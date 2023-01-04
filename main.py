# todo: list of things still need to be done
#  - 1. Random walk discovery protocol
#    - https://ieeexplore.ieee.org/abstract/document/8885214
#    - Thought: random walk protocol is executed after each round, as the reputation values have changed.
#    - Somehow we need reciprocity in MeritRank or incentives for edge creation
#    ---- maybe we should do it simple? Broadcast your new edge weights every round. Don't care about package loss. ----
#    - MeritRank
#       - Reciprocity?
#       - Incentives for accepting new strangers? really necessary? benevolent users want to increase overall accuracy
#       - Stranger protocol
#       - Random walk strategy in decentralized networks
#       - underlying rep mechanism
#  - 2. Reputation system
#    -
#  - 3. Bristle implementation
#  - 4. Reputation mechanism which updates your own reputation values on edges


# todo today
#  - 1. implement simple random walk protocol as discussed with Rohan. extra thought, what if we give random walks a maximum length, then we can perform transitivity decay locally afterwards.
#  - 2. implement simple MeritRank only using connectivity decay
#  - 3. start with simple Bristle implementation

if __name__ == '__main__':
    print('Hello world')
