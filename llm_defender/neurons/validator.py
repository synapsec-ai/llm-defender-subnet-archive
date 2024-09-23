"""
Validator docstring here
"""

# Import standard modules
import asyncio
import hashlib
import secrets
import sys
import time
import traceback
from argparse import ArgumentParser
from uuid import uuid4
import random

# Import custom modules
import bittensor as bt
import numpy as np

# Import subnet modules
import llm_defender.base as LLMDefenderBase
from llm_defender.core import validator as LLMDefenderCore


def update_metagraph(validator: LLMDefenderCore.SubnetValidator) -> None:
    try:
        validator.metagraph = asyncio.run(
            validator.sync_metagraph(validator.metagraph, validator.subtensor)
        )
        validator.neuron_logger(
            severity="INFOX",
            message=f"Metagraph synced: {validator.metagraph}"
        )
    except TimeoutError as e:
        validator.neuron_logger(
            severity="ERROR",
            message=f"Metagraph sync timed out: {e}"
        )


async def update_metagraph_async(validator: LLMDefenderCore.SubnetValidator) -> None:
    await asyncio.to_thread(update_metagraph, validator)


def update_and_check_hotkeys(validator: LLMDefenderCore.SubnetValidator) -> None:
    validator.check_hotkeys()
    if validator.wallet.hotkey.ss58_address not in validator.metagraph.hotkeys:
        validator.neuron_logger(
            severity="ERROR",
            message=f"Hotkey is not registered on metagraph: {validator.wallet.hotkey.ss58_address}."
        )


async def update_and_check_hotkeys_async(
    validator: LLMDefenderCore.SubnetValidator,
) -> None:
    await asyncio.to_thread(update_and_check_hotkeys, validator)


def save_validator_state(validator: LLMDefenderCore.SubnetValidator) -> None:
    validator.save_state()


async def save_validator_state_async(
    validator: LLMDefenderCore.SubnetValidator,
) -> None:
    await asyncio.to_thread(save_validator_state, validator)


def save_miner_state(validator: LLMDefenderCore.SubnetValidator):
    validator.save_miner_state()


async def save_miner_state_async(validator: LLMDefenderCore.SubnetValidator):
    await asyncio.to_thread(save_miner_state, validator)


def truncate_miner_state(
    validator: LLMDefenderCore.SubnetValidator, max_number_of_responses_per_miner: int
):
    validator.truncate_miner_state(max_number_of_responses_per_miner)


async def truncate_miner_state_async(
    validator: LLMDefenderCore.SubnetValidator, max_number_of_responses_per_miner: int
):
    await asyncio.to_thread(
        truncate_miner_state, validator, max_number_of_responses_per_miner
    )


def save_used_nonces(validator: LLMDefenderCore.SubnetValidator):
    validator.save_used_nonces()


async def save_used_nonces_async(validator: LLMDefenderCore.SubnetValidator):
    await asyncio.to_thread(save_used_nonces, validator)


def query_axons(synapse_uuid, uids_to_query, validator):
    # Sync implementation
    # Broadcast query to valid Axons
    nonce = secrets.token_hex(24)
    timestamp = str(int(time.time()))
    data_to_sign = (
        f"{synapse_uuid}{nonce}{validator.wallet.hotkey.ss58_address}{timestamp}"
    )
    # query['analyzer'] = "Sensitive Information"
    responses = validator.dendrite.query(
        uids_to_query,
        LLMDefenderBase.SubnetProtocol(
            analyzer=validator.query["analyzer"],
            subnet_version=validator.subnet_version,
            synapse_uuid=synapse_uuid,
            synapse_signature=LLMDefenderBase.sign_data(
                hotkey=validator.wallet.hotkey, data=data_to_sign, log_level=validator.log_level
            ),
            synapse_nonce=nonce,
            synapse_timestamp=timestamp,
        ),
        timeout=validator.timeout,
        deserialize=True,
    )
    return responses


def send_notification_synapse(
    synapse_uuid, validator, axons_with_valid_ip, prompt_to_analyze
):
    encoded_prompt = prompt_to_analyze.get("prompt").encode("utf-8")
    prompt_hash = hashlib.sha256(encoded_prompt).hexdigest()
    nonce = secrets.token_hex(24)
    timestamp = str(int(time.time()))
    data_to_sign = (
        f"{synapse_uuid}{nonce}{validator.wallet.hotkey.ss58_address}{timestamp}"
    )
    validator.neuron_logger(
        severity="TRACEX",
        message=f"Sent notification synapse to: {axons_with_valid_ip} with encoded prompt: {encoded_prompt} for prompt: {prompt_to_analyze}."
    )
    responses = validator.dendrite.query(
        axons_with_valid_ip,
        LLMDefenderBase.SubnetProtocol(
            subnet_version=validator.subnet_version,
            synapse_uuid=synapse_uuid,
            synapse_signature=LLMDefenderBase.sign_data(
                hotkey=validator.wallet.hotkey, data=data_to_sign, log_level=validator.log_level
            ),
            synapse_nonce=nonce,
            synapse_timestamp=timestamp,
            synapse_hash=prompt_hash,
        ),
        timeout=(validator.timeout / 2),
        deserialize=True,
    )
    return responses


def score_unused_axons(validator, uids_not_to_query):
    # Process UIDs we did not query (set scores to 0)
    for uid in uids_not_to_query:
        validator.neuron_logger(
            severity="TRACE",
            message=f"Setting score for not queried UID: {uid}. Old score: {validator.scores[uid]}"
        )
        validator.scores[uid] = 0.99 * validator.scores[uid]
        validator.neuron_logger(
            severity="TRACE",
            message=f"Set score for not queried UID: {uid}. New score: {validator.scores[uid]}"
        )


async def score_unused_axons_async(validator, uids_not_to_query):
    await asyncio.to_thread(score_unused_axons, validator, uids_not_to_query)


def handle_empty_responses(validator, list_of_uids):
    # This must be SYNC process, because we need to wait until the subnetwork syncs
    # Handle all responses empty
    validator.neuron_logger(
        severity="INFO",
        message="Received empty response from all miners"
    )
    # If we receive empty responses from all axons, we can just set the scores to none for all the uids we queried
    score_unused_axons(validator, list_of_uids)
    validator.neuron_logger(
        severity="INFO",
        message=f"Sleeping for: {bt.__blocktime__/3} seconds"
    )
    time.sleep(bt.__blocktime__ / 3)


def format_responses(
    validator, list_of_uids, responses, synapse_uuid, prompt_to_analyze
):
    # Process the responses
    response_data, responses_invalid_uids, responses_valid_uids = validator.process_responses(
        query=prompt_to_analyze,
        processed_uids=list_of_uids,
        responses=responses,
        synapse_uuid=synapse_uuid,
    )
    return response_data, responses_invalid_uids, responses_valid_uids


def handle_invalid_prompt(validator):
    # This must be SYNC process
    # If we cannot get a valid prompt, sleep for a moment and retry the loop
    validator.neuron_logger(
        severity="WARNING",
        message=f"Unable to get a valid query from the Prompt API, received: {validator.query}. Please report this to subnet developers if the issue persists."
    )

    # Sleep and retry
    validator.neuron_logger(
        severity="ERROR",
        message=f"Sleeping for: {bt.__blocktime__/3} seconds before retrying the loop."
    )
    time.sleep(bt.__blocktime__ / 3)


def attach_response_to_validator(validator, response_data):
    for res in response_data:
        hotkey = res["hotkey"]

        if hotkey not in validator.miner_responses:
            validator.miner_responses[hotkey] = [res]
        else:
            validator.miner_responses[hotkey].append(res)


def update_weights(validator: LLMDefenderCore.SubnetValidator):
    # Periodically update the weights on the Bittensor blockchain.
    try:
        is_validator_healthy, health_data = validator.healthcheck_api.get_health()
        if not is_validator_healthy:
            validator.neuron_logger(
                severity="ERROR",
                message=f'Validator is not healthy. Cant set weights. Health data: {health_data}'
            )
        else:
            asyncio.run(validator.commit_weights())

            # Update validators knowledge of the last updated block
            if not validator.debug_mode:
                validator.last_updated_block = validator.subtensor.get_current_block()
    except TimeoutError as e:
        validator.neuron_logger(
            severity="ERROR", 
            message=f"Committing weights timed out: {e}"
        )


async def update_weights_async(validator):
    await asyncio.to_thread(update_weights, validator)


async def get_average_score_per_analyzer(validator):

    results = {}

    for hotkey, response_list in validator.miner_responses.items():
        
        if not response_list:
            validator.neuron_logger(
                severity="DEBUGX",
                message=f"Response history for miner: {hotkey} is empty: {response_list}"
            )
            continue
        
        analyzer_scores = {}
        weights = {}
        missed_responses = {}
        successful_responses = {}
        
        for response in response_list:

            analyzer = response["analyzer"]

            score = response["scored_response"]["scores"]["total_analyzer_raw"]
            weight = response["weight"]

            if analyzer not in analyzer_scores:
                analyzer_scores[analyzer] = []
            if analyzer not in weights:
                weights[analyzer] = []
            if analyzer not in missed_responses:
                missed_responses[analyzer] = 1
            if analyzer not in successful_responses:
                successful_responses[analyzer] = 1

            analyzer_scores[analyzer].append(score)
            weights[analyzer].append(weight)
            if not response['engine_data']:
                missed_responses[analyzer] += 1
            else:
                successful_responses[analyzer] += 1
        
        weighted_averages = {}
        missed_response_ratios = {}
        
        for key in missed_responses:
            total = missed_responses[key] + successful_responses[key]
            if total > 0:
                missed_response_ratios[key] = missed_responses[key] / total
            else: 
                missed_response_ratios[key] = 1.0

        for key in analyzer_scores:
            scores = analyzer_scores[key]
            weight = weights[key]
            
            preprocessed_missed_response_penalty = 1 - missed_response_ratios[key]
            if preprocessed_missed_response_penalty >= 0.95:
                missed_response_penalty = 1.0 
            else:
                missed_response_penalty = preprocessed_missed_response_penalty
            
            weighted_sum = sum(score * w for score, w in zip(scores, weight))
            total_weight = sum(weight)
            
            weighted_average = (weighted_sum * missed_response_penalty) / total_weight
            weighted_averages[key] = weighted_average
        
        # Store the results using hotkey as the key
        results[hotkey] = weighted_averages
        
    return results


async def main(validator: LLMDefenderCore.SubnetValidator):
    """
    This function executes the main function for the validator.
    """

    # Get module version
    version = LLMDefenderBase.config["module_version"]

    # Step 7: The Main Validation Loop
    validator.neuron_logger(
        severity="INFO", message=f"Starting validator loop with version: {version}"
    )
    validator.healthcheck_api.append_metric(metric_name="neuron_running", value=True)

    while True:
        try:
            # ensure that the number of responses per miner is below a number
            max_number_of_responses_per_miner = 100
            truncate_miner_state(validator, max_number_of_responses_per_miner)

            # Periodically sync subtensor status and save the state file
            if validator.step % 5 == 0:
                await update_metagraph_async(validator)
                await update_and_check_hotkeys_async(validator)
                await asyncio.gather(
                    save_validator_state_async(validator),
                    save_miner_state_async(validator),
                )
            if validator.step % 20 == 0:
                await asyncio.gather(
                    save_used_nonces_async(validator),
                )

            # Get all axons
            all_axons = validator.metagraph.axons
            validator.neuron_logger(
                severity="TRACE", 
                message=f"All axons: {all_axons}"
            )
            # If there are more axons than scores, append the scores list
            if len(validator.metagraph.uids.tolist()) > len(validator.scores):
                validator.neuron_logger(
                    severity="INFO", 
                    message=f"Discovered new Axons, current scores: {validator.scores}"
                )
                additional_zeros = np.zeros(
                    (len(validator.metagraph.uids.tolist()) - len(validator.scores)),
                    dtype=np.float32,
                )
                validator.scores = np.concatenate((validator.scores, additional_zeros))
                validator.neuron_logger(
                    severity="INFO", 
                    message=f"Updated scores, new scores: {validator.scores}"
                )

            # if there are more axons than prompt_injection_scores, append the prompt_injection_scores list
            if len(validator.metagraph.uids.tolist()) > len(
                validator.prompt_injection_scores
            ):
                validator.neuron_logger(
                    severity="INFO", 
                    message=f"Discovered new Axons, current prompt_injection_scores: {validator.prompt_injection_scores}"
                )
                additional_zeros = np.zeros(
                    (
                        len(validator.metagraph.uids.tolist())
                        - len(validator.prompt_injection_scores)
                    ),
                    dtype=np.float32,
                )
                validator.prompt_injection_scores = np.concatenate(
                    (validator.prompt_injection_scores, additional_zeros)
                )
                validator.neuron_logger(
                    severity="INFO", 
                    message=f"Updated prompt_injection_scores, new prompt_injection_scores: {validator.prompt_injection_scores}"
                )

            # if there are more axons than sensitive_information_socres, append the sensitive_information_scores list
            if len(validator.metagraph.uids.tolist()) > len(
                validator.sensitive_information_scores
            ):
                validator.neuron_logger(
                    severity="INFO", 
                    message=f"Discovered new Axons, current scores: {validator.scores}"
                )
                additional_zeros = np.zeros(
                    (
                        len(validator.metagraph.uids.tolist())
                        - len(validator.sensitive_information_scores)
                    ),
                    dtype=np.float32,
                )
                validator.sensitive_information_scores = np.concatenate(
                    (validator.sensitive_information_scores, additional_zeros)
                )
                validator.neuron_logger(
                    severity="INFO", 
                    message=f"Updated sensitive_information_scores, new sensitive_information_scores: {validator.sensitive_information_scores}"
                )

            axons_with_valid_ip = validator.determine_valid_axons(all_axons)
            # miner_hotkeys_to_broadcast = [valid_ip_axon.hotkey for valid_ip_axon in axons_with_valid_ip]

            if not axons_with_valid_ip:
                validator.neuron_logger(
                    severity="WARNING", 
                    message="No axons with valid IPs found")
                validator.neuron_logger(
                    severity="DEBUG", 
                    message=f"Sleeping for: {bt.__blocktime__/3} seconds")
                time.sleep(bt.__blocktime__ / 3)
                continue

            # Generate prompt to be analyzed by the miners
            synapse_uuid = str(uuid4())
            analyzer = random.choice(["Prompt Injection", "Sensitive Information"])
            prompt_to_analyze = validator.serve_prompt(
                analyzer=analyzer
            )

            validator.neuron_logger(
                severity="INFOX", 
                message=f"Serving prompt: {prompt_to_analyze} for analyzer: {analyzer}"
            )

            is_prompt_invalid = (
                prompt_to_analyze is None
                or "analyzer" not in prompt_to_analyze.keys()
                or "label" not in prompt_to_analyze.keys()
                or "weight" not in prompt_to_analyze.keys()
            )
            if is_prompt_invalid:
                handle_invalid_prompt(validator)
                continue

            # Get list of UIDs to send the payload synapse
            (uids_to_query, list_of_uids, uids_not_to_query) = (
                await validator.get_uids_to_query_async(all_axons=all_axons)
            )

            validator.healthcheck_api.append_metric(metric_name="axons.total_filtered_axons", value = len(uids_not_to_query))
            validator.healthcheck_api.append_metric(metric_name="axons.total_queried_axons", value = len(uids_to_query))

            if not uids_to_query:
                validator.neuron_logger(
                    severity="WARNING",
                    message=f"UIDs to query is empty: {uids_to_query}")
                continue

            validator.neuron_logger(
                severity="INFO", 
                message=f"Sending Payload Synapse to {len(uids_to_query)} targets starting with UID: {list_of_uids[0]} and ending with UID: {list_of_uids[-1]}"
            )

            responses = await validator.send_payload_message(
                synapse_uuid=synapse_uuid,
                uids_to_query=uids_to_query,
                prompt_to_analyze=prompt_to_analyze,
                timeout=validator.timeout,
            )
            # await score_unused_axons_async(validator, uids_not_to_query)

            # are_responses_empty = all(item.output is None for item in responses)
            # if are_responses_empty:
            # handle_empty_responses(validator, list_of_uids)
            # continue

            validator.neuron_logger(
                severity="TRACE", 
                message=f"Received responses: {responses}"
            )

            response_data, responses_invalid_uids, responses_valid_uids = format_responses(
                validator, list_of_uids, responses, synapse_uuid, prompt_to_analyze
            )
            attach_response_to_validator(validator, response_data)

            # Print stats
            validator.neuron_logger(
                severity="DEBUG", 
                message=f"Processed UIDs: {list(list_of_uids)}"
            )

            current_block = validator.subtensor.get_current_block()
            validator.neuron_logger(
                severity="DEBUG", 
                message=f"Current step: {validator.step}. Current block: {current_block}. Last updated block: {validator.last_updated_block}"
            )

            # Calculate analyzer average scores, calculate overall scores and then set weights
            if current_block - validator.last_updated_block > 100:
                averages = await get_average_score_per_analyzer(validator)
                
                for hotkey, data in averages.items():

                    uid = validator.hotkeys.index(hotkey)
                    
                    data_keys = [k for k in data]

                    if "Prompt Injection" in data_keys:
                        validator.prompt_injection_scores[uid] = data[
                            "Prompt Injection"
                        ]
                    else:
                        validator.prompt_injection_scores[uid] = 0.0

                    if "Sensitive Information" in data_keys:
                        validator.sensitive_information_scores[uid] = data[
                            "Sensitive Information"
                        ]
                    else:
                        validator.sensitive_information_scores[uid] = 0.0

                validator.neuron_logger(
                    severity="DEBUG", 
                    message=f"Prompt Injection Analyzer scores: {validator.prompt_injection_scores}"
                )
                validator.neuron_logger(
                    severity="DEBUG", 
                    message=f"Sensitive Information Analyzer scores: {validator.sensitive_information_scores}"
                )

                validator.determine_overall_scores()
                # Commit/set weights (depending on if commit reveal is enabled or not)
                await update_weights_async(validator)

            # Reveal weights (if enabled)
            if validator.subtensor.get_subnet_hyperparameters(netuid=validator.neuron_config.netuid).commit_reveal_weights_enabled:
                try: 
                    validator.reveal_weights_in_queue()
                except TimeoutError as e:
                    validator.neuron_logger(
                        severity="ERROR",
                        message=f"Reveal weights timed out: {e}"
                    )

            # End the current step and prepare for the next iteration.
            validator.healthcheck_api.update_rates()
            if validator.target_group == 0:
                validator.healthcheck_api.append_metric(metric_name='iterations', value=1)

            validator.neuron_logger(
                severity="SUCCESS", 
                message=f"Processed {len(list_of_uids)} neurons out of which {len(responses_valid_uids)} provided valid response and {len(responses_invalid_uids)} invalid response."
            )
            validator.step += 1

            # Sleep for a duration equivalent to 1/3 of the block time (i.e., time between successive blocks).
            validator.neuron_logger(
                severity="DEBUG", 
                message=f"Sleeping for: {bt.__blocktime__/3} seconds"
            )
            time.sleep(bt.__blocktime__ / 3)

        # If we encounter an unexpected error, log it for debugging.
        except RuntimeError as e:
            validator.neuron_logger(
                severity="ERROR",
                message=e
            )
            traceback.print_exc()

        # If the user interrupts the program, gracefully exit.
        except KeyboardInterrupt:
            validator.neuron_logger(
                severity="SUCCESS", 
                message="Keyboard interrupt detected. Exiting validator.")
            sys.exit()

        except Exception as e:
            validator.neuron_logger(
                severity="ERROR",
                message=e
            )
            traceback.print_exc()


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    # Parse command line arguments
    parser = ArgumentParser()
    
    parser.add_argument("--netuid", type=int, default=14, help="The chain subnet uid.")

    parser.add_argument(
        "--load_state",
        type=str,
        default="True",
        help="WARNING: Setting this value to False clears the old state.",
    )

    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Running the validator in debug mode ignores selected validity checks. Not to be used in production.",
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["INFO", "INFOX", "DEBUG", "DEBUGX", "TRACE", "TRACEX"],
        help="Determine the logging level used by the subnet modules",
    )

    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default="http://prompt-generation-api:8000/v1",
        help="Determine the vLLM base url used for the prompt generation",
    )

    parser.add_argument(
        "--vllm_api_key",
        type=str,
        default="default_api_key",
        help="Determine the vLLM api key used for the prompt generation",
    )
    parser.add_argument(
        "--vllm_model_name",
        type=str,
        default="synapsecai/mixtral-8x7b-instruct-llm-defender-GPTQ-v0.1",
        help="Determines the vLLM model to utilize",
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

    # Create a validator based on the Class definitions and initialize it
    subnet_validator = LLMDefenderCore.SubnetValidator(parser=parser)
    if (
        not subnet_validator.apply_config(
            bt_classes=[bt.subtensor, bt.logging, bt.wallet]
        )
        or not subnet_validator.initialize_neuron()
    ):
        subnet_validator.neuron_logger(
            severity="ERROR", 
            message="Unable to initialize Validator. Exiting.")
        sys.exit()

    asyncio.run(main(subnet_validator))
