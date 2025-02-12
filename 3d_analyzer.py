import streamlit as st
import trimesh
import numpy as np
import stable_baselines3 as sb3
from stable_baselines3 import PPO
import pyvista as pv
import gym
from gym import spaces
import os

# Check for missing dependencies
try:
    import open3d as o3d
except OSError:
    st.warning("Open3D could not be loaded. Ensure 'libGL.so.1' is installed.")

def load_model(file):
    try:
        model = trimesh.load_mesh(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def analyze_model(model):
    metadata = {
        "vertices": len(model.vertices),
        "faces": len(model.faces),
        "edges": len(model.edges),
        "is_watertight": model.is_watertight,
        "bounding_box": model.bounding_box.extents.tolist()
    }
    return metadata

def suggest_improvements(metadata):
    suggestions = []
    if metadata["faces"] > 50000:
        suggestions.append("Reduce poly count for better performance.")
    if not metadata["is_watertight"]:
        suggestions.append("Ensure the model is watertight to avoid rendering issues.")
    return suggestions

class ModelOptimizationEnv(gym.Env):
    def __init__(self, metadata):
        super(ModelOptimizationEnv, self).__init__()
        self.metadata = metadata
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)  # Example: Adjust poly count and UV map
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(len(metadata),), dtype=np.float32)

    def step(self, action):
        self.metadata["faces"] *= (1 - action[0])  # Example: Reduce poly count
        reward = -self.metadata["faces"]  # Reward for reducing complexity
        return np.array(list(self.metadata.values())), reward, True, {}

    def reset(self):
        return np.array(list(self.metadata.values()))

def train_rl_model(metadata):
    env = ModelOptimizationEnv(metadata)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    return model

def apply_rl_optimization(model, metadata):
    env = ModelOptimizationEnv(metadata)
    obs = env.reset()
    action, _ = model.predict(obs)
    optimized_metadata, _, _, _ = env.step(action)
    return dict(zip(metadata.keys(), optimized_metadata))

def visualize_model(model):
    if not hasattr(model, 'vertices') or not hasattr(model, 'faces'):
        st.warning("Model visualization is not available.")
        return
    
    plotter = pv.Plotter()
    vertices = np.array(model.vertices)
    faces = np.hstack([[len(f)] + list(f) for f in model.faces])
    mesh = pv.PolyData(vertices, faces)
    plotter.add_mesh(mesh, color="white")
    plotter.show()

def main():
    st.title("3D Model Analyzer & Game-Ready Suggestions")
    uploaded_file = st.file_uploader("Upload 3D Model (.obj, .fbx, .stl, .gltf)", type=["obj", "fbx", "stl", "gltf"])
    
    if uploaded_file is not None:
        model = load_model(uploaded_file)
        if model:
            metadata = analyze_model(model)
            st.subheader("Model Metadata")
            st.json(metadata)
            
            st.subheader("Suggested Improvements")
            suggestions = suggest_improvements(metadata)
            for suggestion in suggestions:
                st.write(f"- {suggestion}")
            
            st.subheader("Applying Reinforcement Learning Optimization")
            rl_model = train_rl_model(metadata)
            optimized_metadata = apply_rl_optimization(rl_model, metadata)
            st.json(optimized_metadata)
            
            st.subheader("3D Model Visualization")
            visualize_model(model)
            
if __name__ == "__main__":
    main()
