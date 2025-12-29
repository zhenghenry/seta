# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from camel.logger import get_logger
from camel.messages import OpenAIMessage
from camel.types import ChatCompletion
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.engine.rollout.verl_engine import VerlEngine

# Import the parent class
from external.camel.camel.models.openai_compatible_model import (
    OpenAICompatibleModel,
)

logger = get_logger(__name__)


class rLLMEngineModel(OpenAICompatibleModel):
    r"""OpenAI-compatible model backend that uses VerlEngine for inference.

    This class wraps the VerlEngine to provide an OpenAI-compatible interface,
    allowing it to be used as a drop-in replacement for OpenAI models in the
    CAMEL framework during RL training with rLLM.

    Args:
        engine (VerlEngine): The VerlEngine instance to use for inference.
        model_type (Union[ModelType, str]): Model identifier/name.
        model_config_dict (Optional[Dict[str, Any]], optional): Configuration
            dictionary for model parameters. (default: :obj:`None`)
        **kwargs (Any): Additional arguments (ignored, for compatibility).
    """

    def __init__(
        self,
        engine: VerlEngine,
        model_type: Union[str, Any],
        model_config_dict: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # Store the engine
        self.engine = engine

        # Initialize parent with minimal required params
        # We don't need actual OpenAI clients since we're using VerlEngine
        super().__init__(
            model_type=model_type,
            model_config_dict=model_config_dict or {},
            api_key="dummy",  # Not used
            url="dummy",  # Not used
            **kwargs,
        )

    def _convert_model_output_to_openai_format(
        self,
        model_output: ModelOutput,
        model_name: str,
    ) -> ChatCompletion:
        """Convert VerlEngine ModelOutput to OpenAI ChatCompletion format.

        Args:
            model_output (ModelOutput): The output from VerlEngine.
            model_name (str): The model name to include in the response.

        Returns:
            ChatCompletion: OpenAI-formatted completion response.
        """
        # Generate a unique completion ID
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        # Build the message content and tool_calls
        message_content = model_output.content if model_output.content else None
        tool_calls_list = None

        if model_output.tool_calls:
            # Convert rLLM ToolCall objects to OpenAI format
            tool_calls_list = []
            for idx, tool_call in enumerate(model_output.tool_calls):
                tool_calls_list.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                    }
                )

        # Construct the message
        message = {
            "role": "assistant",
            "content": message_content,
            "refusal": None,
            "annotations": [],
            "audio": None,
            "function_call": None,
        }

        if tool_calls_list:
            message["tool_calls"] = tool_calls_list

        # Build the choice
        choice = {
            "finish_reason": model_output.finish_reason,
            "index": 0,
            "logprobs": None,
            "message": message,
        }

        # Build usage statistics
        usage = {
            "completion_tokens": model_output.completion_length,
            "prompt_tokens": model_output.prompt_length,
            "total_tokens": model_output.prompt_length
            + model_output.completion_length,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {
                "audio_tokens": 0,
                "cached_tokens": 0,
            },
        }

        # Construct the full ChatCompletion response
        response = ChatCompletion(
            id=completion_id,
            choices=[choice],
            created=int(time.time()),
            model=model_name,
            object="chat.completion",
            service_tier=None,
            system_fingerprint=None,
            usage=usage,
        )

        return response

    async def _arequest_chat_completion(
        self,
        messages: List[OpenAIMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        """Async request chat completion using VerlEngine.

        Args:
            messages (List[OpenAIMessage]): Message list in OpenAI format.
            tools (Optional[List[Dict[str, Any]]]): Tool schemas.

        Returns:
            ChatCompletion: OpenAI-formatted completion response.
            ModelOutput: Original VerlEngine model output.
        """
        # Extract any additional kwargs from model_config_dict
        request_config = self.model_config_dict.copy()
        request_config.pop("stream", None)  # VerlEngine doesn't support streaming

        # Convert OpenAI messages to dict format expected by VerlEngine
        messages_dict = [dict(msg) for msg in messages]

        # Prepare kwargs for VerlEngine
        engine_kwargs = {}
        if tools:
            engine_kwargs["tools"] = tools

        # Add other relevant config
        if "temperature" in request_config:
            engine_kwargs["temperature"] = request_config["temperature"]
        if "top_p" in request_config:
            engine_kwargs["top_p"] = request_config["top_p"]
        if "top_k" in request_config:
            engine_kwargs["top_k"] = request_config["top_k"]
        if "max_tokens" in request_config:
            engine_kwargs["max_tokens"] = request_config["max_tokens"]

        # Call VerlEngine's get_model_response
        model_output: ModelOutput = await self.engine.get_model_response(
            messages=messages_dict, **engine_kwargs
        )

        # Convert to OpenAI format
        return (
            self._convert_model_output_to_openai_format(
            model_output=model_output, model_name=str(self.model_type)
            ),
            model_output
        )

    async def _arequest_parse(
        self,
        messages: List[OpenAIMessage],
        response_format: Type[BaseModel],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        """Async request with structured output parsing using VerlEngine.

        Args:
            messages (List[OpenAIMessage]): Message list in OpenAI format.
            response_format (Type[BaseModel]): Pydantic model for response.
            tools (Optional[List[Dict[str, Any]]]): Tool schemas.

        Returns:
            ChatCompletion: OpenAI-formatted completion response with parsed output.
        """
        raise NotImplementedError(
            "_arequest_parse is not implemented yet for VerlEngine."
        )

    def _request_parse(
        self,
        messages: List[OpenAIMessage],
        response_format: Type[BaseModel],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        """Sync version of _request_parse (not supported - VerlEngine is async only).

        Args:
            messages (List[OpenAIMessage]): Message list in OpenAI format.
            response_format (Type[BaseModel]): Pydantic model for response.
            tools (Optional[List[Dict[str, Any]]]): Tool schemas.

        Raises:
            NotImplementedError: VerlEngine only supports async operations.
        """
        raise NotImplementedError(
            "VerlEngine only supports async operations. Use _arequest_parse instead."
        )

    def _request_chat_completion(
        self,
        messages: List[OpenAIMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        """Sync version of _request_chat_completion (not supported).

        Args:
            messages (List[OpenAIMessage]): Message list in OpenAI format.
            tools (Optional[List[Dict[str, Any]]]): Tool schemas.

        Raises:
            NotImplementedError: VerlEngine only supports async operations.
        """
        raise NotImplementedError(
            "VerlEngine only supports async operations. "
            "Use _arequest_chat_completion instead."
        )

    @property
    def stream(self) -> bool:
        """VerlEngine does not support streaming.

        Returns:
            bool: Always False.
        """
        return False