__all__ = ['extract_model_features', 'analyze_topology', 'analyze_uv_maps', 'get_game_readiness_score']

from typing import Dict, Any, List, Optional, Tuple
import streamlit as st
import trimesh
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
import io
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from model_env import ModelOptimizationEnv
from rl_agent import DQNAgent
from llm_assistant import ModelAssistant

class ModelQualityPredictor(nn.Module):
    def __init__(self):
        super(ModelQualityPredictor, self).__init__()
        # Input features: vertex count, face count, uv metrics, etc.
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

def extract_model_features(mesh):
    """Extract numerical features from the 3D model for ML analysis"""
    features = {
        'vertex_count': len(mesh.vertices),
        'face_count': len(mesh.faces),
        'volume': mesh.volume if hasattr(mesh, 'volume') else 0,
        'surface_area': mesh.area,
        'is_watertight': float(mesh.is_watertight),
        'is_winding_consistent': float(mesh.is_winding_consistent),
        'euler_number': len(mesh.vertices) - len(mesh.faces) + len(mesh.edges),
        'bbox_volume': np.prod(mesh.extents),
        'compactness': mesh.area ** (3/2) / (36 * np.pi * mesh.volume ** 2) if mesh.volume > 0 else 0,
        # Calculate edge lengths manually using vertex positions
        'edge_length_mean': calculate_edge_lengths(mesh)
    }
    return features

def calculate_edge_lengths(mesh):
    """Calculate mean edge length for a mesh"""
    try:
        # Get unique edges
        edges = mesh.edges_unique
        # Get vertices for each edge
        edge_vertices = mesh.vertices[edges]
        # Calculate edge lengths using numpy
        lengths = np.linalg.norm(edge_vertices[:, 1] - edge_vertices[:, 0], axis=1)
        return float(np.mean(lengths)) if len(lengths) > 0 else 0
    except:
        return 0

def analyze_uv_maps(mesh):
    """Analyze UV mapping quality and coverage"""
    if not hasattr(mesh, 'visual') or not hasattr(mesh.visual, 'uv'):
        return {
            'has_uvs': False,
            'coverage': 0,
            'overlapping': False,
            'suggestions': ['Model lacks UV mapping. Consider adding UV coordinates.']
        }
    
    uv_coords = mesh.visual.uv
    
    # Calculate UV coverage
    uv_area = np.abs(np.cross(uv_coords[1:] - uv_coords[0], uv_coords[2:] - uv_coords[0])).sum() / 2
    coverage = min(uv_area / 1.0, 1.0)  # Normalize to [0,1]
    
    # Check for overlapping UVs (simplified)
    overlapping = len(np.unique(uv_coords, axis=0)) < len(uv_coords)
    
    # More detailed UV analysis
    uv_bounds = np.array([uv_coords.min(axis=0), uv_coords.max(axis=0)])
    uv_size = uv_bounds[1] - uv_bounds[0]
    
    suggestions = []
    if coverage < 0.8:
        suggestions.append('UV coverage is low. Consider repacking UVs to maximize texture space.')
    if overlapping:
        suggestions.append('Detected overlapping UVs. Consider unwrapping affected areas.')
    if np.any(uv_size > 1.0):
        suggestions.append('UVs extend beyond 0-1 space. Consider normalizing UV coordinates.')
    
    return {
        'has_uvs': True,
        'coverage': coverage,
        'overlapping': overlapping,
        'uv_size': uv_size.tolist(),
        'suggestions': suggestions
    }

def analyze_topology(mesh):
    """Analyze mesh topology and provide suggestions"""
    suggestions = []
    
    # Basic topology checks
    vertex_count = len(mesh.vertices)
    face_count = len(mesh.faces)
    
    if vertex_count > 10000:
        suggestions.append(f'High vertex count ({vertex_count:,}). Consider decimation for game optimization.')
    
    if hasattr(mesh, 'is_watertight') and not mesh.is_watertight:
        suggestions.append('Mesh has holes or non-manifold edges. Fix for proper game engine import.')
    
    # Calculate face density
    try:
        area = mesh.area
        density = face_count / area if area > 0 else 0
        if density > 1000:  # arbitrary threshold
            suggestions.append('High polygon density detected. Consider optimization.')
    except:
        density = 0
    
    # Check for separate components using connected components
    try:
        components = mesh.split() if hasattr(mesh, 'split') else [mesh]
        part_count = len(components)
        if part_count > 1:
            suggestions.append(f'Mesh consists of {part_count} separate parts. Consider unifying or organizing parts.')
    except:
        part_count = 1
    
    # Check for degenerate faces using face areas
    try:
        face_areas = mesh.area_faces
        degenerate_count = np.sum(face_areas < 1e-8)
        if degenerate_count > 0:
            suggestions.append(f'Found {degenerate_count} potentially degenerate faces. Clean up geometry.')
    except:
        degenerate_count = 0
    
    # Check for T-junctions
    try:
        t_junctions = find_t_junctions(mesh)
        if (t_junctions):
            suggestions.append(f'Found {len(t_junctions)} T-junctions. Consider welding vertices.')
    except:
        t_junctions = []
    
    return {
        'vertex_count': vertex_count,
        'face_count': face_count,
        'is_watertight': mesh.is_watertight if hasattr(mesh, 'is_watertight') else False,
        'density': density,
        'part_count': part_count,
        't_junctions': t_junctions,  # Add T-junctions to the results
        'suggestions': suggestions
    }

def find_t_junctions(mesh):
    """Detect T-junctions in the mesh"""
    t_junctions = []
    
    # Get all vertices and edges
    vertices = mesh.vertices
    edges = mesh.edges_unique
    
    # Create a spatial index for efficient vertex lookup
    from rtree import index
    
    # Create R-tree index
    idx = index.Index()
    for i, v in enumerate(vertices):
        idx.insert(i, (v[0], v[1], v[2], v[0], v[1], v[2]))
    
    # Check each edge
    for edge in edges:
        v1 = vertices[edge[0]]
        v2 = vertices[edge[1]]
        
        # Get the midpoint of the edge
        midpoint = (v1 + v2) / 2
        
        # Find nearby vertices
        nearby_points = list(idx.intersection((
            min(v1[0], v2[0]), min(v1[1], v2[1]), min(v1[2], v2[2]),
            max(v1[0], v2[0]), max(v1[1], v2[1]), max(v1[2], v2[2])
        )))
        
        # Check if any vertex lies on the edge (within a small threshold)
        edge_vector = v2 - v1
        edge_length = np.linalg.norm(edge_vector)
        
        for point_idx in nearby_points:
            if point_idx in edge:  # Skip edge endpoints
                continue
                
            point = vertices[point_idx]
            # Calculate distance from point to edge
            v = point - v1
            projection = np.dot(v, edge_vector) / edge_length
            
            if 0 < projection < edge_length:
                # Calculate perpendicular distance
                dist = np.linalg.norm(v - projection * edge_vector / edge_length)
                
                # If point is very close to edge, it's a T-junction
                if dist < 1e-6:  # Adjustable threshold
                    t_junctions.append({
                        'vertex': point_idx,
                        'edge': edge,
                        'position': point
                    })
    
    return t_junctions

def get_game_readiness_score(analysis_results: Dict[str, Any]) -> float:
    """Calculate game readiness score based on analysis results"""
    score = 100.0
    
    # Topology penalties
    if not analysis_results['topology']['is_watertight']:
        score -= 15
    
    if analysis_results['topology']['vertex_count'] > 10000:
        score -= 10
    
    # UV penalties
    if not analysis_results['uv_analysis']['has_uvs']:
        score -= 20
    elif analysis_results['uv_analysis']['has_overlap']:
        score -= 10
    
    # Density penalties
    if analysis_results['topology']['density'] > 1000:
        score -= 10
    
    # Ensure score stays within 0-100 range
    return max(0, min(100, score))

def create_3d_visualization(mesh, analysis_results):
    """Create an interactive 3D visualization of the model with annotations"""
    
    # Create the base mesh visualization with better defaults
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Calculate vertex normals for better shading
    vertex_colors = np.ones((len(vertices), 3)) * 0.8  # Light gray base color
    
    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=0.9,
            colorscale='Viridis',
            intensity=vertices[:, 2],
            lighting=dict(
                ambient=0.6,
                diffuse=0.8,
                fresnel=0.2,
                specular=0.5,
                roughness=0.4
            ),
            lightposition=dict(x=100, y=100, z=100),
            hoverinfo='text',
            hovertext=['Mesh Surface'],
            showscale=False
        )
    ])
    
    # Add detailed markers for issues
    if not analysis_results['topology']['is_watertight']:
        edges = mesh.edges_unique
        edge_points = mesh.vertices[edges]
        midpoints = (edge_points[:, 0] + edge_points[:, 1]) / 2
        
        hover_texts = [
            f"Non-manifold Edge<br>" +
            f"Location: ({x:.2f}, {y:.2f}, {z:.2f})<br>" +
            "Issue: Mesh not watertight here"
            for x, y, z in midpoints
        ]
        
        fig.add_trace(go.Scatter3d(
            x=midpoints[:, 0],
            y=midpoints[:, 1],
            z=midpoints[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='diamond',
                line=dict(color='darkred', width=2)
            ),
            name='Non-manifold Edges',
            hoverinfo='text',
            hovertext=hover_texts,
            showlegend=True
        ))
    
    # Enhanced high-density area visualization
    if analysis_results['topology']['density'] > 1000:
        face_centers = np.mean(mesh.vertices[mesh.faces], axis=1)
        face_areas = mesh.area_faces
        dense_faces = face_areas < np.median(face_areas) / 2
        
        if np.any(dense_faces):
            dense_centers = face_centers[dense_faces]
            hover_texts = [
                f"High Density Area<br>" +
                f"Location: ({x:.2f}, {y:.2f}, {z:.2f})<br>" +
                f"Face Area: {area:.6f}"
                for (x, y, z), area in zip(dense_centers, face_areas[dense_faces])
            ]
            
            fig.add_trace(go.Scatter3d(
                x=dense_centers[:, 0],
                y=dense_centers[:, 1],
                z=dense_centers[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color='yellow',
                    symbol='circle',
                    line=dict(color='orange', width=1)
                ),
                name='High-density Areas',
                hoverinfo='text',
                hovertext=hover_texts,
                showlegend=True
            ))
    # Enhanced T-junction visualization
    if analysis_results['topology']['t_junctions']:
        t_junction_positions = np.array([tj['position'] for tj in analysis_results['topology']['t_junctions']])
        hover_texts = [
            f"T-Junction<br>" +
            f"Location: ({x:.2f}, {y:.2f}, {z:.2f})<br>" +
            "Issue: Vertex causing T-junction"
            for x, y, z in t_junction_positions
        ]
        
        fig.add_trace(go.Scatter3d(
            x=t_junction_positions[:, 0],
            y=t_junction_positions[:, 1],
            z=t_junction_positions[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color='blue',
                symbol='x',
                line=dict(color='darkblue', width=2)
            ),
            name='T-junctions',
            hoverinfo='text',
            hovertext=hover_texts,
            showlegend=True
        ))
    
    # Enhanced layout with better viewport
    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            zaxis=dict(showgrid=False, zeroline=False),
            bgcolor='rgb(33, 33, 33)'
        ),
        title=dict(
            text="Interactive 3D Model Analysis",
            font=dict(size=24, color='white')
        ),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(51, 51, 51, 0.8)',
            bordercolor='rgba(255, 255, 255, 0.3)',
            borderwidth=1,
            font=dict(color='white')
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return fig

def optimize_model(mesh):
    """Optimize 3D model using RL agent"""
    env = ModelOptimizationEnv(mesh)
    agent = DQNAgent(state_size=10, action_size=6)
    
    max_episodes = 50
    best_mesh = None
    best_score = float('-inf')
    
    for episode in range(max_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            total_reward += reward
            state = next_state
            
            if done:
                if total_reward > best_score:
                    best_score = total_reward
                    best_mesh = env.mesh.copy()
                break
    
    return best_mesh if best_mesh is not None else mesh, best_score

def run_optimization_with_explanation(mesh, initial_analysis):
    """Run optimization with detailed explanation of the process"""
    st.write("🤖 Starting Optimization Process...")
    
    report = {
        'steps': [],
        'decisions': [],
        'improvements': [],
        'learning_process': [],
        'final_metrics': {}
    }
    
    with st.expander("🧠 Model's Thought Process", expanded=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        env = ModelOptimizationEnv(mesh)
        agent = DQNAgent(state_size=10, action_size=6)
        
        for episode in range(50):
            progress_bar.progress(episode/50)
            state = env.reset()
            
            episode_report = {
                'episode': episode,
                'actions_taken': [],
                'reasoning': [],
                'rewards': []
            }
            
            while True:
                # Get action and explanation
                action, reasoning = agent.act_with_explanation(state)
                next_state, reward, done, info = env.step(action)
                
                episode_report['actions_taken'].append(action)
                episode_report['reasoning'].append(reasoning)
                episode_report['rewards'].append(reward)
                
                status_text.write(f"Episode {episode}: {reasoning}")
                
                if done:
                    break
            
            report['steps'].append(episode_report)
    
    return report

def display_optimization_report(report):
    """Display the detailed optimization report"""
    st.subheader("📊 Optimization Analysis Report")
    
    # Display overall improvements
    st.write("### Key Improvements")
    for improvement in report['improvements']:
        st.write(f"• {improvement}")
    
    # Show learning process
    st.write("### 🧠 Learning Process")
    for step in report['steps'][-5:]:  # Show last 5 episodes
        with st.expander(f"Episode {step['episode']}"):
            for action, reason, reward in zip(
                step['actions_taken'], 
                step['reasoning'], 
                step['rewards']
            ):
                st.write(f"**Action:** {action}")
                st.write(f"**Reasoning:** {reason}")
                st.write(f"**Reward:** {reward}")

def display_model_assistant(mesh, analysis_results):
    """Interactive AI assistant for model analysis"""
    st.write("### 💬 Model Assistant")
    st.write("Ask me anything about the analyzed 3D model!")
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat input
    user_question = st.text_input("Your question:", key="model_question")
    
    if user_question:
        # Generate response based on model analysis
        response = generate_model_response(
            user_question, 
            mesh, 
            analysis_results,
            st.session_state.chat_history
        )
        
        # Add to chat history
        st.session_state.chat_history.append({
            "user": user_question,
            "assistant": response
        })
    
    # Display chat history
    for chat in st.session_state.chat_history:
        st.write(f"👤 **You:** {chat['user']}")
        st.write(f"🤖 **Assistant:** {chat['assistant']}")
def generate_model_response(question, mesh, analysis_results, chat_history):
    """Generate natural language response about the model"""
    # Here you can integrate with an actual LLM API
    # For now, we'll use a simple template-based approach
    
    context = {
        'vertex_count': len(mesh.vertices),
        'face_count': len(mesh.faces),
        'is_watertight': analysis_results['topology']['is_watertight'],
        'has_uvs': analysis_results['uv_analysis']['has_uvs'],
        'suggestions': analysis_results['topology']['suggestions']
    }
    
    # Basic response generation based on keywords
    if 'topology' in question.lower():
        return f"The model has {context['vertex_count']} vertices and {context['face_count']} faces. " \
               f"It is {'watertight' if context['is_watertight'] else 'not watertight'}."
    
    if 'uv' in question.lower():
        return f"The model {'has' if context['has_uvs'] else 'does not have'} UV mapping."
    
    if 'improve' in question.lower() or 'optimize' in question.lower():
        return "Based on my analysis, here are the suggested improvements:\n" + \
               "\n".join([f"• {s}" for s in context['suggestions']])
    
    return "I'm sorry, I don't understand the question. Try asking about the model's topology, UV mapping, or optimization suggestions."

def display_detailed_analysis(analysis_results: Dict[str, Any]):
    """Display detailed analysis of the 3D model"""
    # Topology Analysis
    st.write("### 🔍 Topology Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Mesh Statistics:**")
        st.write(f"- Vertices: {analysis_results['topology']['vertex_count']:,}")
        st.write(f"- Faces: {analysis_results['topology']['face_count']:,}")
        st.write(f"- Parts: {analysis_results['topology']['part_count']}")
        st.write(f"- Density: {analysis_results['topology']['density']:.2f}")
        st.write(f"- Watertight: {'✅' if analysis_results['topology']['is_watertight'] else '❌'}")
    
    with col2:
        st.write("**Topology Issues:**")
        if analysis_results['topology']['suggestions']:
            for suggestion in analysis_results['topology']['suggestions']:
                st.write(f"- {suggestion}")
        else:
            st.write("No topology issues found! ✅")
    
    # UV Analysis
    st.write("### 🎨 UV Analysis")
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("**UV Map Status:**")
        st.write(f"- Has UVs: {'✅' if analysis_results['uv_analysis']['has_uvs'] else '❌'}")
        if analysis_results['uv_analysis']['has_uvs']:
            st.write(f"- UV Coverage: {analysis_results['uv_analysis']['uv_coverage']:.1f}%")
            st.write(f"- UV Overlap: {'Yes ⚠️' if analysis_results['uv_analysis']['has_overlap'] else 'No ✅'}")
    
    with col4:
        st.write("**UV Issues:**")
        if analysis_results['uv_analysis']['suggestions']:
            for suggestion in analysis_results['uv_analysis']['suggestions']:
                st.write(f"- {suggestion}")
        else:
            st.write("No UV issues found! ✅")
    
    # Feature Analysis
    st.write("### 📊 Model Features")
    features = analysis_results['features']
    if features:
        col5, col6 = st.columns(2)
        with col5:
            st.write("**Geometric Features:**")
            st.write(f"- Volume: {features.get('volume', 0):.2f}")
            st.write(f"- Surface Area: {features.get('surface_area', 0):.2f}")
            st.write(f"- Bounding Box Volume: {features.get('bbox_volume', 0):.2f}")
        
        with col6:
            st.write("**Quality Metrics:**")
            st.write(f"- Compactness: {features.get('compactness', 0):.2f}")
            st.write(f"- Edge Length (mean): {features.get('edge_length_mean', 0):.4f}")
            st.write(f"- Euler Number: {features.get('euler_number', 0)}")

def main():
    # Set page config
    st.set_page_config(
        page_title="3D Model Analyzer",
        page_icon="🎮",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better UI
    st.markdown("""
        <style>
        .main {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            border-radius: 5px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .reportview-container {
            background-color: #1E1E1E;
        }
        .sidebar .sidebar-content {
            background-color: #262626;
        }
        h1 {
            color: #4CAF50;
            text-align: center;
            padding: 20px;
        }
        h2 {
            color: #90EE90;
        }
        h3 {
            color: #98FB98;
        }
        .element-container {
            background-color: #262626;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create sidebar for controls (simplified)
    with st.sidebar:
        st.image("https://your-logo-url.png", use_column_width=True)  # Fixed deprecated parameter
        st.title("Controls")
        
        st.write("---")
        st.markdown("### Viewport Controls")
        st.markdown("""
        - 🔄 Rotate: Click and drag
        - 🔍 Zoom: Scroll wheel
        - ✋ Pan: Right-click + drag
        - 📱 Touch: Two fingers
        - 🎯 Reset: Double-click
        """)

    # Main content
    st.title("3D Model Analyzer for Game Development")
    
    # File upload (only in main content)
    uploaded_file = st.file_uploader(
        "Upload your 3D model to get analysis and optimization suggestions", 
        type=['obj', 'stl', 'ply', 'glb', 'fbx'],
        help="Supported formats: OBJ, STL, PLY, GLB, FBX"
    )

    if uploaded_file is not None:
        try:
            # Load and initial analysis
            mesh = trimesh.load(io.BytesIO(uploaded_file.getvalue()), 
                              file_type=uploaded_file.name.split('.')[-1])
            
            # Initial analysis
            analysis_results = {
                'uv_analysis': analyze_uv_maps(mesh),
                'topology': analyze_topology(mesh),
                'features': extract_model_features(mesh)
            }
            game_readiness = get_game_readiness_score(analysis_results)
            
            # Create tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Analysis", 
                "🔍 3D View", 
                "🛠️ Optimization",
                "🤖 AI Assistant"
            ])
            
            with tab1:
                # Metrics display
                col1, col2, col3 = st.columns([2,3,2])
                with col1:
                    st.metric("Game Readiness Score", f"{game_readiness:.1f}%")
                with col2:
                    st.metric("Vertex Count", f"{analysis_results['topology']['vertex_count']:,}")
                with col3:
                    st.metric("Face Count", f"{analysis_results['topology']['face_count']:,}")
                
                # Detailed analysis
                display_detailed_analysis(analysis_results)
            
            with tab2:
                fig = create_3d_visualization(mesh, analysis_results)
                st.plotly_chart(fig, use_container_width=True, key="main_3d_view")
                
                st.info("""
                **Interaction Instructions:**
                - Rotate: Click and drag
                - Zoom: Scroll or pinch
                - Pan: Right-click and drag
                - Reset View: Double-click
                - Hover over markers to see issue details
                """)
            
            with tab3:
                if st.button("Start Optimization", key="optimize_button"):
                    optimization_report = run_optimization_with_explanation(mesh, analysis_results)
                    display_optimization_report(optimization_report)
            
            with tab4:
                st.header("AI Model Assistant")
                st.write("Ask me anything about your 3D model!")
                
                # Initialize session state for chat
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                if 'assistant' not in st.session_state:
                    st.session_state.assistant = ModelAssistant()
                
                # Chat input
                user_question = st.text_input(
                    "Your question:",
                    key="model_question",
                    placeholder="e.g., What are the main issues with my model?"
                )
                
                if user_question:
                    # Generate response
                    response = st.session_state.assistant.generate_response(
                        user_question,
                        mesh,
                        analysis_results,
                        st.session_state.chat_history
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "user": user_question,
                        "assistant": response
                    })
                
                # Display chat history in reverse order (newest first)
                for chat in reversed(st.session_state.chat_history):
                    with st.container():
                        st.markdown(f"**You:** {chat['user']}")
                        st.markdown(f"**Assistant:** {chat['assistant']}")
                        st.markdown("---")
                
                # Example questions
                with st.expander("📝 Example Questions"):
                    st.markdown("""
                    - What are the main issues with my model?
                    - How can I optimize the polygon count?
                    - Is this model suitable for mobile games?
                    - What's causing the non-manifold edges?
                    - How can I fix the UV mapping issues?
                    """)

        except Exception as e:
            st.error(f"Error processing the model: {str(e)}")
            return

if __name__ == '__main__':
    main()

class DQNAgent:
    # ... existing code ...
    
    def act_with_explanation(self, state):
        """Take action and provide explanation"""
        action = self.act(state)
        
        # Generate explanation for the action
        explanations = []
        if action[0] > 0.5:
            explanations.append("Decided to reduce polygon count due to high vertex density")
        if action[1] > 0.5:
            explanations.append("Attempting to optimize UV layout for better texture utilization")
        if action[2] > 0.5:
            explanations.append("Welding vertices to fix mesh topology issues")
        
        explanation = " and ".join(explanations) if explanations else "Observing current state"
        
        return action, explanation