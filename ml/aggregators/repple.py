from ml.aggregators.foolsgold import FoolsGold


class Repple(FoolsGold):
    def requires_gossip(self) -> bool:
        return True
