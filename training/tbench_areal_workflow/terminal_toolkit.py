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
import atexit
import os
import subprocess
import threading
import time
import uuid
import shlex
import select
from queue import Queue, Empty
from typing import Any, Dict, List, Optional, Tuple


from camel.logger import get_logger
from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.toolkits import TerminalToolkit
from camel.utils import MCPServer

logger = get_logger(__name__)

# Try to import docker, but don't make it a hard requirement
try:
    import docker
    from docker.errors import NotFound, APIError
    from docker.models.containers import Container
except ImportError:
    docker = None
    NotFound = None
    APIError = None
    Container = None

@MCPServer()
class TerminalToolkitTrace(TerminalToolkit):
    """
    A toolkit for LLM agents to execute and interact with terminal commands
    in either a local or a sandboxed Docker environment.

    Args:
    use_docker_backend (bool): If True, all commands are executed in a
        Docker container. Defaults to False.
    docker_container_name (Optional[str]): The name of the Docker
        container to use. Required if use_docker_backend is True.
    working_dir (str): The base directory for all operations.
        For the local backend, this acts as a security sandbox.
    session_logs_dir (Optional[str]): The directory to store session
        logs. Defaults to a 'terminal_logs' subfolder in the
        working_dir.
    timeout (int): The default timeout in seconds for blocking
        commands. Defaults to 60.

    Note:

    """
    