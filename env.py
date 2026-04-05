import numpy as np

class PacManEnv:
    def __init__(self, grid_size=20, n_pacmen=2, n_ghosts=2, max_steps=200, power_duration=10):
        self.grid_size = grid_size
        self.n_pacmen = n_pacmen
        self.n_ghosts = n_ghosts
        self.n_agents = n_pacmen + n_ghosts
        
        self.max_steps = max_steps
        self.power_duration = power_duration
        
        self.action_space_size = 5 # Up, Down, Left, Right, Stay
        # Obs space: 6 channels (Walls, Food, Power, Pacmen, Ghosts, Self) + 1 (Power mode status)
        self.obs_dim = 6 * grid_size * grid_size + 1
        
        self.reset()
        
    def reset(self):
        self.step_cnt = 0
        self.power_timer = 0
        self.team_lives = 3
        
        self.pacmen_pos = []
        self.ghosts_pos = []
        self.pacmen_alive = [True] * self.n_pacmen
        
        # 0: Empty, 1: Wall, 2: Food, 3: Power
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Build walls (simple borders + some random inner blocks)
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        
        # Add some random static walls
        np.random.seed(42) # Fixed seed for walls for consistency
        for _ in range(self.grid_size * 2):
            rx, ry = np.random.randint(2, self.grid_size-2, size=2)
            self.grid[rx, ry] = 1
            
        np.random.seed() # reset seed
        

        # Spawn 2 power pellets
        empty_cells = np.argwhere(self.grid == 0)
        np.random.shuffle(empty_cells)
        for i in range(2):
            px, py = empty_cells[i]
            self.grid[px, py] = 3
            
        # Spawn agents
        empty_cells = empty_cells[2:]
        for i in range(self.n_pacmen):
            self.pacmen_pos.append(list(empty_cells[i]))
        empty_cells = empty_cells[self.n_pacmen:]
            
        for i in range(self.n_ghosts):
            self.ghosts_pos.append(list(empty_cells[i]))
            
        # Spawn food on ALL remaining valid empty spaces
        for x in range(1, self.grid_size-1):
            for y in range(1, self.grid_size-1):
                if self.grid[x, y] == 0:
                    self.grid[x, y] = 2
            
        return self._get_obs()

    def _get_obs(self):
        obs = []
        # Build Base channels
        c_walls = (self.grid == 1).astype(np.float32)
        c_food = (self.grid == 2).astype(np.float32)
        c_power = (self.grid == 3).astype(np.float32)
        
        c_pacmen = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for i, (px, py) in enumerate(self.pacmen_pos):
            if self.pacmen_alive[i]:
                c_pacmen[px, py] = 1.0
                
        c_ghosts = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for gx, gy in self.ghosts_pos:
            c_ghosts[gx, gy] = 1.0

        pm_status = 1.0 if self.power_timer > 0 else 0.0

        for i in range(self.n_pacmen):
            c_self = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            if self.pacmen_alive[i]:
                c_self[self.pacmen_pos[i][0], self.pacmen_pos[i][1]] = 1.0
            
            channels = np.stack([c_walls, c_food, c_power, c_pacmen, c_ghosts, c_self])
            flat_obs = np.concatenate([channels.flatten(), [pm_status]])
            obs.append(flat_obs)
            
        for i in range(self.n_ghosts):
            c_self = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            c_self[self.ghosts_pos[i][0], self.ghosts_pos[i][1]] = 1.0
            
            channels = np.stack([c_walls, c_food, c_power, c_pacmen, c_ghosts, c_self])
            flat_obs = np.concatenate([channels.flatten(), [pm_status]])
            obs.append(flat_obs)
            
        return obs
    
    def _move(self, pos, action):
        # 0: Up, 1: Down, 2: Left, 3: Right, 4: Stay
        nx, ny = pos[0], pos[1]
        if action == 0: nx -= 1
        elif action == 1: nx += 1
        elif action == 2: ny -= 1
        elif action == 3: ny += 1
        
        # Check Wall collision
        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.grid[nx, ny] != 1:
            return [nx, ny]
        return pos

    def step(self, actions):
        self.step_cnt += 1
        rewards = np.zeros(self.n_agents)
        
        old_ghosts_pos = [p[:] for p in self.ghosts_pos]
        old_pacmen_pos = [p[:] for p in self.pacmen_pos]

        # Process Pac-Men moves
        for i in range(self.n_pacmen):
            if self.pacmen_alive[i]:
                new_pos = self._move(self.pacmen_pos[i], actions[i])
                if new_pos == self.pacmen_pos[i]:
                    rewards[i] -= 2.0 # Penalty for hitting wall or standing still
                else:
                    rewards[i] += 0.5 # Reward for successful organic motion
                    
                self.pacmen_pos[i] = new_pos
                px, py = self.pacmen_pos[i]
                
                # Check food
                if self.grid[px, py] == 2:
                    self.grid[px, py] = 0
                    rewards[i] += 10.0
                # Check power pellet
                elif self.grid[px, py] == 3:
                    self.grid[px, py] = 0
                    rewards[i] += 30.0
                    self.power_timer = self.power_duration
                    
        # Process Ghost moves
        for i in range(self.n_ghosts):
            a_idx = self.n_pacmen + i
            new_pos = self._move(self.ghosts_pos[i], actions[a_idx])
            
            if new_pos == self.ghosts_pos[i]:
                rewards[a_idx] -= 2.0 # Penalty for ghost standing still
            else:
                rewards[a_idx] += 0.5 # Reward for ghost organic motion
                
            self.ghosts_pos[i] = new_pos
            
            # Distance to nearest pacman reward
            min_dist = 999
            for j in range(self.n_pacmen):
                if self.pacmen_alive[j]:
                    dist = abs(self.ghosts_pos[i][0] - self.pacmen_pos[j][0]) + abs(self.ghosts_pos[i][1] - self.pacmen_pos[j][1])
                    if dist < min_dist: min_dist = dist
                    
            old_min_dist = 999
            for j in range(self.n_pacmen):
                if self.pacmen_alive[j]:
                    dist = abs(old_ghosts_pos[i][0] - old_pacmen_pos[j][0]) + abs(old_ghosts_pos[i][1] - old_pacmen_pos[j][1])
                    if dist < old_min_dist: old_min_dist = dist
                    
            if min_dist < old_min_dist:
                rewards[a_idx] += 0.1
        
        # Check Collisions 
        is_power_mode = self.power_timer > 0
        for i in range(self.n_pacmen):
            if not self.pacmen_alive[i]: continue
            for j in range(self.n_ghosts):
                if self.pacmen_pos[i] == self.ghosts_pos[j]:
                    g_idx = self.n_pacmen + j
                    if is_power_mode:
                        # Pacman eats ghost
                        rewards[i] += 20.0
                        rewards[g_idx] -= 20.0
                        # Respawn ghost
                        empty_cells = np.argwhere(self.grid == 0)
                        if len(empty_cells) > 0:
                            rx, ry = empty_cells[np.random.randint(len(empty_cells))]
                            self.ghosts_pos[j] = [rx, ry]
                    else:
                        # Ghost eats pacman
                        rewards[i] -= 20.0
                        rewards[g_idx] += 20.0
                        self.team_lives -= 1
                        
                        if self.team_lives <= 0:
                            self.pacmen_alive = [False] * self.n_pacmen
                            for k in range(self.n_pacmen):
                                self.pacmen_pos[k] = [-1, -1]
                        else:
                            empty_cells = np.argwhere(self.grid == 0)
                            if len(empty_cells) > 0:
                                rx, ry = empty_cells[np.random.randint(len(empty_cells))]
                                self.pacmen_pos[i] = [rx, ry]
                        
        if self.power_timer > 0:
            self.power_timer -= 1
            
        dones = [False] * self.n_agents
        
        # Episode ends if max steps reached
        if self.step_cnt >= self.max_steps:
            dones = [True] * self.n_agents
            
        # Episode ends if all pacmen are dead
        if not any(self.pacmen_alive):
            dones = [True] * self.n_agents
            
        # Episode ends if no food and no power pellets left
        if not np.any(self.grid == 2) and not np.any(self.grid == 3):
            dones = [True] * self.n_agents
            
        # Send per-agent done flags to make it easier for MADDPG structure
        return self._get_obs(), rewards.tolist(), dones, {}

    def get_eval_state(self):
        # Extract a clean dictionary state for visualizer
        return {
            "grid": self.grid.copy(),
            "pacmen_pos": [p[:] for p in self.pacmen_pos],
            "pacmen_alive": self.pacmen_alive[:],
            "ghosts_pos": [p[:] for p in self.ghosts_pos],
            "power_mode": self.power_timer > 0
        }
