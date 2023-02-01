from asyncio import ensure_future, get_event_loop

from ipv8.configuration import WalkerDefinition, Strategy, default_bootstrap_defs, ConfigBuilder
from ipv8.types import Peer
from ipv8_service import IPv8

from experiment_settings.settings import Settings
from federated_learning.community import FLCommunity, Role


async def start_communities(amount: int):
    # Start the server
    builder = ConfigBuilder().clear_keys().clear_overlays()
    builder.add_key("Server", "medium", "../keys/ecserver.pem")
    # We provide the 'started' function to the 'on_start'.
    # We will call the overlay's 'started' function without any
    # arguments once IPv8 is initialized.
    builder.add_overlay("ReppleCommunity", "Server", [WalkerDefinition(Strategy.RandomWalk, 100, {'timeout': 3.0})],
                        default_bootstrap_defs, {}, [])
    server = IPv8(builder.finalize(), extra_communities={'ReppleCommunity': FLCommunity})
    s: Peer = server.keys['Server']
    await server.start()

    nodes = list()
    for i in range(1, amount + 1):
        builder = ConfigBuilder().clear_keys().clear_overlays()
        builder.add_key(f"peer {i}", "medium", f"../keys/ec{i}.pem")
        # We provide the 'started' function to the 'on_start'.
        # We will call the overlay's 'started' function without any
        # arguments once IPv8 is initialized.
        builder.add_overlay("ReppleCommunity", f"peer {i}", [WalkerDefinition(Strategy.RandomWalk, 100, {'timeout': 3.0})],
                            default_bootstrap_defs, {}, [('started',)])
        node = IPv8(builder.finalize(), extra_communities={'ReppleCommunity': FLCommunity})
        nodes.append(node)
        await node.start()

    server.overlays[0].start(settings, -1, Role.SERVER, None)
    for i, n in enumerate(nodes):
        n.overlays[0].start(settings, i, Role.NODE, s)


if __name__ == '__main__':
    settings = Settings()
    ensure_future(start_communities(2))
    get_event_loop().run_forever()
