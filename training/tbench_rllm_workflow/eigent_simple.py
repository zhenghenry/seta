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

import asyncio
import datetime
import os
import platform
import uuid

from camel.logger import get_logger
from camel.messages.base import BaseMessage
from camel.models import BaseModelBackend, ModelFactory
from camel.societies.workforce import Workforce
from camel.tasks.task import Task
from camel.toolkits import (
    AgentCommunicationToolkit,
    NoteTakingToolkit,
    TerminalToolkit,
    ToolkitMessageIntegration,
)
from camel.types import ModelPlatformType, ModelType
from camel.utils.commons import api_keys_required

logger = get_logger(__name__)


from rllm_chat_agent import rLLMChatAgent as ChatAgent

from prompts import get_developer_agent_prompt, get_coordinator_agent_prompt, get_task_agent_prompt, get_new_worker_prompt

def send_message_to_user(
    message_title: str,
    message_description: str,
    message_attachment: str = "",
) -> str:
    r"""Use this tool to send a tidy message to the user, including a
    short title, a one-sentence description, and an optional attachment.

    This one-way tool keeps the user informed about your progress,
    decisions, or actions. It does not require a response.
    You should use it to:
    - Announce what you are about to do.
      For example:
      message_title="Starting Task"
      message_description="Searching for papers on GUI Agents."
    - Report the result of an action.
      For example:
      message_title="Search Complete"
      message_description="Found 15 relevant papers."
    - Report a created file.
      For example:
      message_title="File Ready"
      message_description="The report is ready for your review."
      message_attachment="report.pdf"
    - State a decision.
      For example:
      message_title="Next Step"
      message_description="Analyzing the top 10 papers."
    - Give a status update during a long-running task.

    Args:
        message_title (str): The title of the message.
        message_description (str): The short description.
        message_attachment (str): The attachment of the message,
            which can be a file path or a URL.

    Returns:
        str: Confirmation that the message was successfully sent.
    """
    print(f"\nAgent Message:\n{message_title} " f"\n{message_description}\n")
    if message_attachment:
        print(message_attachment)
    logger.info(
        f"\nAgent Message:\n{message_title} "
        f"{message_description} {message_attachment}"
    )
    return (
        f"Message successfully sent to user: '{message_title} "
        f"{message_description} {message_attachment}'"
    )


def developer_agent_factory(
    model: BaseModelBackend,
    task_id: str = None,
    terminal_toolkit_kwargs: dict = None,
    system: str = platform.system(),
    machine: str = platform.machine(),
    is_workforce: bool = False,
    working_directory: str = "CAMEL_WORKDIR",
):
    r"""Factory for creating a developer agent."""
    # Initialize message integration
    message_integration = ToolkitMessageIntegration(
        message_handler=send_message_to_user
    )

    # Initialize toolkits
    # terminal_toolkit = TerminalToolkit(safe_mode=True, clone_current_env=False)
    terminal_toolkit = TerminalToolkit(**terminal_toolkit_kwargs)
    note_toolkit = NoteTakingToolkit(working_directory=working_directory)

    # Add messaging to toolkits
    terminal_toolkit = message_integration.register_toolkits(terminal_toolkit)
    note_toolkit = message_integration.register_toolkits(note_toolkit)

    # Get enhanced tools
    tools = [
        *terminal_toolkit.get_tools()[0:-1], # Use without ask human tool
        *note_toolkit.get_tools(),
    ]

    system_message = get_developer_agent_prompt(
                                                current_date = str(datetime.date.today()), 
                                                system = system, 
                                                machine = machine, 
                                                is_workforce = is_workforce
                                                )

    return ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Developer Agent",
            content=system_message,
        ),
        model=model,
        tools=tools,
    )



async def main(
  task_instruction:str,
  terminal_toolkit_kwargs: dict = None,
  logdir: str = "",
  system: str = platform.system(),
  machine: str = platform.machine(),
  working_directory: str = f"CAMEL_WORKDIR_{uuid.uuid4().hex}",
):
    # Ensure working directory exists
    os.makedirs(working_directory, exist_ok=True)

    # Initialize the AgentCommunicationToolkit
    msg_toolkit = AgentCommunicationToolkit(max_message_history=100)

    # Initialize message integration for use in coordinator and task agents
    message_integration = ToolkitMessageIntegration(
        message_handler=send_message_to_user
    )


    # Create a single model backend for all agents
    model_backend = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4_1,
        model_config_dict={
            "stream": False,
        },
    )

    model_backend_reason = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4_1,
        model_config_dict={
            "stream": False,
        },
    )

    task_id = 'workforce_task'

    # Create custom agents for the workforce
    coordinator_agent = ChatAgent(
        system_message=(
            get_coordinator_agent_prompt(
                current_date = str(datetime.date.today()), 
                system = system, 
                machine = machine, 
            )
        ),
        model=model_backend_reason,
        tools=[
            *NoteTakingToolkit(
                working_directory=working_directory
            ).get_tools(),
        ],
    )
    task_agent = ChatAgent(
        get_task_agent_prompt(
            current_date = str(datetime.date.today()), 
            system = system, 
            machine = machine,
        ),
        model=model_backend_reason,
        tools=[
            *NoteTakingToolkit(
                working_directory=working_directory
            ).get_tools(),
        ],
    )
    new_worker_agent = ChatAgent(
        get_new_worker_prompt(),
        model=model_backend,
        tools=[
            *message_integration.register_toolkits(
                NoteTakingToolkit(working_directory=working_directory)
            ).get_tools(),
        ],
    )

    # Create agents using factory functions
    developer_agent = developer_agent_factory(
        model_backend_reason,
        task_id,
        terminal_toolkit_kwargs,
        system=system,
        machine=machine,
        is_workforce=True,
        working_directory=working_directory,
    )


    msg_toolkit.register_agent("Worker", new_worker_agent)
    msg_toolkit.register_agent("Developer_Agent", developer_agent)

    # Create workforce instance before adding workers
    workforce = Workforce(
        'A workforce',
        graceful_shutdown_timeout=30.0,  # 30 seconds for debugging
        share_memory=False,
        coordinator_agent=coordinator_agent,
        task_agent=task_agent,
        new_worker_agent=new_worker_agent,
        use_structured_output_handler=False,
        task_timeout_seconds=900.0,
    )

    workforce.add_single_agent_worker(
        "Developer Agent: A master-level coding assistant with a powerful "
        "terminal. It can write and execute code, manage files, automate "
        "desktop tasks, and deploy web applications to solve complex "
        "technical challenges.",
        worker=developer_agent,
    )

    # specify the task to be solved
    human_task = Task(
        content=(
            f"""
{task_instruction}
            """
        ),
        id='0',
    )

    # Use the async version directly to avoid hanging with async tools
    await workforce.process_task_async(human_task)

    # Test WorkforceLogger features
    print("\n--- Workforce Log Tree ---")
    print(workforce.get_workforce_log_tree())

    print("\n--- Workforce KPIs ---")
    kpis = workforce.get_workforce_kpis()
    for key, value in kpis.items():
        print(f"{key}: {value}")

    log_file_path = f"{logdir}eigent_logs.json"
    print(f"\n--- Dumping Workforce Logs to {log_file_path} ---")
    workforce.dump_workforce_logs(log_file_path)
    print(f"Logs dumped. Please check the file: {log_file_path}")


if __name__ == "__main__":
    asyncio.run(main())
