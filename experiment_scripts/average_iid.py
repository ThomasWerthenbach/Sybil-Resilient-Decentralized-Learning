from asyncio import ensure_future, get_event_loop

from ipv8.configuration import WalkerDefinition, Strategy, default_bootstrap_defs, ConfigBuilder
from ipv8_service import IPv8

from community.base_community import BaseCommunity
from experiment_settings.settings import Settings


async def start_communities(amount: int):
    nodes = list()
    for i in range(1, amount + 1):
        builder = ConfigBuilder().clear_keys().clear_overlays()
        builder.add_key("my peer", "medium", f"../keys/ec{i}.pem")
        # We provide the 'started' function to the 'on_start'.
        # We will call the overlay's 'started' function without any
        # arguments once IPv8 is initialized.
        builder.add_overlay("ReppleCommunity", "my peer", [WalkerDefinition(Strategy.RandomWalk, 10, {'timeout': 3.0})],
                            default_bootstrap_defs, {}, [('started',)])
        node = IPv8(builder.finalize(), extra_communities={'ReppleCommunity': BaseCommunity})
        nodes.append(node)
        await node.start()

    for i, n in enumerate(nodes):
        n.overlays[0].start(settings, i)


if __name__ == '__main__':
    settings = Settings()
    ensure_future(start_communities(2))
    get_event_loop().run_forever()
