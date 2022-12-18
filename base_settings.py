class Settings:
    def __init__(self):
        self.peers: int = 10
        self.random_walks: int = 10
        self.max_random_walk_length: int = 10
        self.transitivity_decay: float = 0.33