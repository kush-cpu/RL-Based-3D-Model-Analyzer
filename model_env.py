import gym
import numpy as np
from gym import spaces
import trimesh

class ModelOptimizationEnv(gym.Env):
    """Custom Environment for 3D model optimization"""
    def __init__(self, mesh):
        super().__init__()
        self.original_mesh = mesh.copy()
        self.mesh = mesh.copy()
        
        # Define action space (6 continuous actions between 0 and 1)
        self.action_space = spaces.Box(
            low=np.zeros(6),
            high=np.ones(6),
            dtype=np.float32
        )
        
        # Define observation space (10 model features)
        self.observation_space = spaces.Box(
            low=np.zeros(10),
            high=np.inf * np.ones(10),
            dtype=np.float32
        )
        
        self.current_score = None
        
    def step(self, action):
        prev_score = self._get_score()
        
        # Apply actions
        try:
            decimation, uv_repack, vertex_weld, hole_fill, unify, density = action
            
            if decimation > 0.1:  # Apply if reduction > 10%
                target_faces = int(len(self.mesh.faces) * (1 - decimation))
                self.mesh = self.mesh.simplify_quadratic_decimation(target_faces)
            
            if vertex_weld > 0.5:
                self.mesh.vertices, _ = trimesh.remesh.collapse_short_edges(
                    self.mesh.vertices, self.mesh.faces, 0.001
                )
            
            # More optimization actions...
            
        except Exception as e:
            return self._get_observation(), -100, True, {"error": str(e)}
        
        # Calculate reward
        new_score = self._get_score()
        reward = new_score - prev_score
        
        # Check if done
        done = new_score >= 95 or self._check_termination()
        
        return self._get_observation(), reward, done, {}
    
    def reset(self):
        self.mesh = self.original_mesh.copy()
        return self._get_observation()
    
    def _get_observation(self):
        from analyzer import extract_model_features
        features = extract_model_features(self.mesh)
        return np.array(list(features.values()))
    
    def _get_score(self):
        from analyzer import analyze_topology, analyze_uv_maps, get_game_readiness_score
        
        analysis = {
            'topology': analyze_topology(self.mesh),
            'uv_analysis': analyze_uv_maps(self.mesh)
        }
        return get_game_readiness_score(analysis)
    
    def _check_termination(self):
        # Check if mesh is still valid
        return (
            len(self.mesh.vertices) < 100 or  # Too few vertices
            not self.mesh.is_volume or  # Invalid volume
            len(self.mesh.faces) < 50    # Too few faces
        )