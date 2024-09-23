"""
This miner script executes the main loop for the miner and keeps the
miner active in the bittensor network.
"""

import time
from argparse import ArgumentParser
import traceback
import bittensor as bt

import llm_defender.base as LLMDefenderBase
from llm_defender.core import miner as LLMDefenderCore


def main(miner: LLMDefenderCore.SubnetMiner):
    """
    This function executes the main miner loop. The miner is configured
    upon the initialization of the miner. If you want to change the
    miner configuration, please adjust the initialization parameters.
    """

    # Link the miner to the Axon
    axon = bt.axon(wallet=miner.wallet, config=miner.neuron_config)
    miner.neuron_logger(
        severity="INFO",
        message=f"Linked miner to Axon: {axon}"
    )

    # Attach the miner functions to the Axon
    axon.attach(
        forward_fn=miner.analysis_forward,
        blacklist_fn=miner.analysis_blacklist,
        priority_fn=miner.analysis_priority,
    ).attach(
        forward_fn=miner.feedback_forward,
        blacklist_fn=miner.metric_blacklist,
        priority_fn=miner.metric_priority
    )
    miner.neuron_logger(
        severity="INFO",
        message=f"Attached functions to Axon: {axon}"
    )

    # Pass the Axon information to the network
    axon.serve(netuid=miner.neuron_config.netuid, subtensor=miner.subtensor)

    miner.neuron_logger(
        severity="INFO",
        message=f"Axon served on network: {miner.neuron_config.subtensor.chain_endpoint} with netuid: {miner.neuron_config.netuid}"
    )
    # Activate the Miner on the network
    axon.start()
    miner.neuron_logger(
        severity="INFO",
        message=f"Axon started on port: {miner.neuron_config.axon.port}"
    )

    # Step 7: Keep the miner alive
    # This loop maintains the miner's operations until intentionally stopped.
    miner.neuron_logger(
        severity="INFO",
        message="Miner has been initialized and we are connected to the network. Start main loop."
    )

    # Get module version
    version = LLMDefenderBase.config["module_version"]

    # When we init, set last_updated_block to current_block
    miner.last_updated_block = miner.subtensor.get_current_block()
    while True:
        try:
            # Below: Periodically update our knowledge of the network graph.
            if miner.step % 600 == 0:
                miner.neuron_logger(
                    severity="DEBUG",
                    message=f"Syncing metagraph: {miner.metagraph} with subtensor: {miner.subtensor}"
                )

                miner.metagraph.sync(subtensor=miner.subtensor)

                # Check registration status
                if miner.wallet.hotkey.ss58_address not in miner.metagraph.hotkeys:
                    miner.neuron_logger(
                        severity="SUCCESS",
                        message=f"Hotkey is not registered on metagraph: {miner.wallet.hotkey.ss58_address}."
                    )

                # Save used nonces
                miner.save_used_nonces()

                # Clean local data
                # miner.clean_local_storage()

            if miner.step % 60 == 0:
                miner.metagraph = miner.subtensor.metagraph(miner.neuron_config.netuid)
                log = (
                    f"Version:{version} | "
                    f"Step:{miner.step} | "
                    f"Block:{miner.metagraph.block.item()} | "
                    f"Stake:{miner.metagraph.S[miner.miner_uid]} | "
                    f"Rank:{miner.metagraph.R[miner.miner_uid]} | "
                    f"Trust:{miner.metagraph.T[miner.miner_uid]} | "
                    f"Consensus:{miner.metagraph.C[miner.miner_uid] } | "
                    f"Incentive:{miner.metagraph.I[miner.miner_uid]} | "
                    f"Emission:{miner.metagraph.E[miner.miner_uid]}"
                )

                miner.neuron_logger(
                    severity="INFO",
                    message=log
                )

                # Print validator stats
                miner.neuron_logger(
                    severity="DEBUG",
                    message=f"Validator stats: {miner.validator_stats}"
                )
                if miner.wandb_enabled:
                    wandb_logs = [
                        {
                            f"{miner.miner_uid}:{miner.wallet.hotkey.ss58_address}_rank": miner.metagraph.R[
                                miner.miner_uid
                            ].item()
                        },
                        {
                            f"{miner.miner_uid}:{miner.wallet.hotkey.ss58_address}_trust": miner.metagraph.T[
                                miner.miner_uid
                            ].item()
                        },
                        {
                            f"{miner.miner_uid}:{miner.wallet.hotkey.ss58_address}_consensus": miner.metagraph.C[
                                miner.miner_uid
                            ].item()
                        },
                        {
                            f"{miner.miner_uid}:{miner.wallet.hotkey.ss58_address}_incentive": miner.metagraph.I[
                                miner.miner_uid
                            ].item()
                        },
                        {
                            f"{miner.miner_uid}:{miner.wallet.hotkey.ss58_address}_emission": miner.metagraph.E[
                                miner.miner_uid
                            ].item()
                        },
                    ]
                    miner.wandb_handler.set_timestamp()
                    for wandb_log in wandb_logs:
                        miner.wandb_handler.log(data=wandb_log, log_level=miner.log_level)
                    miner.neuron_logger(
                        severity="TRACE",
                        message=f"Wandb logs added: {wandb_logs}"
                    )

            miner.step += 1
            time.sleep(1)

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            axon.stop()
            miner.neuron_logger(
                severity="SUCCESS",
                message="Miner killed by keyboard interrupt."
            )
            if miner.wandb_handler:
                miner.wandb_handler.wandb_run.finish()
            break
        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception:
            miner.neuron_logger(
                severity="SUCCESS",
                message=traceback.format_exc()
            )
            continue


# This is the main function, which runs the miner.
if __name__ == "__main__":
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument("--netuid", type=int, default=14, help="The chain subnet uid")
    parser.add_argument(
        "--logging.logging_dir",
        type=str,
        default="/var/log/bittensor",
        help="Provide the log directory",
    )

    parser.add_argument(
        "--validator_min_stake",
        type=float,
        default=10000.0,
        help="Determine the minimum stake the validator should have to accept requests",
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["INFO", "INFOX", "DEBUG", "DEBUGX", "TRACE", "TRACEX"],
        help="Determine the logging level used by the subnet modules",
    )

    parser.add_argument(
        "--disable_healthcheck",
        action="store_true",
        help="Set this argument if you want to disable the healthcheck API. Enabled by default."
    )

    parser.add_argument(
        "--healthcheck_host",
        type=str,
        default="0.0.0.0",
        help="Set the healthcheck API host. Defaults to 0.0.0.0 to expose it outside of the container.",
    )

    parser.add_argument(
        "--healthcheck_port",
        type=int,
        default=6000,
        help="Determine the port used by the healthcheck API.",
    )

    # Create a miner based on the Class definitions
    subnet_miner = LLMDefenderCore.SubnetMiner(parser=parser)

    main(subnet_miner)
