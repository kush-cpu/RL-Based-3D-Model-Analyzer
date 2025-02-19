import openai
from typing import Dict, List, Any
import os
from dotenv import load_dotenv
from analyzer import get_game_readiness_score  # Add this import
from config import Config

class ModelAssistant:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
    def generate_response(self, 
                         question: str, 
                         mesh_data: Dict[str, Any], 
                         analysis_results: Dict[str, Any],
                         chat_history: List[Dict[str, str]] = None) -> str:
        """Generate a response using OpenAI's API"""
        try:
            # Create system context
            system_context = self._create_system_context(mesh_data, analysis_results)
            
            # Format messages
            messages = [{"role": "system", "content": system_context}]
            
            if chat_history:
                for chat in chat_history[-5:]:
                    messages.append({"role": "user", "content": chat["user"]})
                    messages.append({"role": "assistant", "content": chat["assistant"]})
            
            messages.append({"role": "user", "content": question})
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def _create_system_context(self, mesh_data: Dict[str, Any], analysis_results: Dict[str, Any]) -> str:
        """Create system context for the LLM"""
        context = f"""You are an AI assistant specializing in 3D model analysis for game development.
Current model information:
- Vertices: {len(mesh_data.vertices)}
- Faces: {len(mesh_data.faces)}
- Is watertight: {analysis_results['topology']['is_watertight']}
- Has UV mapping: {analysis_results['uv_analysis']['has_uvs']}
- Game readiness score: {get_game_readiness_score(analysis_results):.1f}%

Identified issues:
{self._format_suggestions(analysis_results)}

Your role is to:
1. Answer questions about the 3D model's quality and structure
2. Explain technical issues in simple terms
3. Provide specific recommendations for improvement
4. Help users understand the optimization process

Please be technical yet approachable in your responses."""
        return context
    
    def _format_suggestions(self, analysis_results: Dict[str, Any]) -> str:
        """Format all suggestions into a readable string"""
        all_suggestions = (
            analysis_results['topology']['suggestions'] +
            analysis_results['uv_analysis']['suggestions']
        )
        return "\n".join([f"- {suggestion}" for suggestion in all_suggestions])