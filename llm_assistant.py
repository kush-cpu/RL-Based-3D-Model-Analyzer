import openai
from typing import Dict, List, Any

class ModelAssistant:
    def __init__(self):
        self.api_key = "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"
        openai.api_key = self.api_key
        
    def generate_response(self, 
                         question: str, 
                         mesh_data: Dict[str, Any], 
                         analysis_results: Dict[str, Any],
                         chat_history: List[Dict[str, str]] = None) -> str:
        """Generate a response using OpenAI's API"""
        
        # Create system context
        system_context = self._create_system_context(mesh_data, analysis_results)
        
        # Format chat history
        messages = [{"role": "system", "content": system_context}]
        
        if chat_history:
            for chat in chat_history[-5:]:  # Include last 5 conversations for context
                messages.append({"role": "user", "content": chat["user"]})
                messages.append({"role": "assistant", "content": chat["assistant"]})
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        try:
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