from asyncio import ensure_future, get_event_loop

from ipv8.configuration import WalkerDefinition, Strategy, default_bootstrap_defs, ConfigBuilder
from ipv8_service import IPv8

from community.ml_community import MLCommunity


async def start_communities(amount: int):
    nodes = list()
    for i in range(1, amount + 1):
        builder = ConfigBuilder().clear_keys().clear_overlays()
        builder.add_key("my peer", "medium", f"keys/ec{i}.pem")
        # We provide the 'started' function to the 'on_start'.
        # We will call the overlay's 'started' function without any
        # arguments once IPv8 is initialized.
        builder.add_overlay("ReppleCommunity", "my peer", [WalkerDefinition(Strategy.RandomWalk, 2, {'timeout': 3.0})],
                            default_bootstrap_defs, {}, [('started',)])
        node = await IPv8(builder.finalize(), extra_communities={'ReppleCommunity': MLCommunity}).start()
        nodes.append(node)

    for i, n in enumerate(nodes):
        n.overlays[0].start(i)



if __name__ == '__main__':
    ensure_future(start_communities(2))
    get_event_loop().run_forever()
