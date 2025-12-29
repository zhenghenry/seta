# 🤓 Terminal-Agent-RL: Terminal Agent evaluation and training based on CAMEL


Note: new TerminalToolkit design document [Terminal Toolkit Design](docs/terminal_toolkit_design.md)

## Getting Started 🎯
### Installation

```bash
# Clone the repository
git clone https://github.com/camel-ai/CAMEL-Terminal-Agent.git
cd CAMEL-Terminal-Agent
bash setup.sh
```

### Run task by task
```bash
#=========================================
# Run single developer agent / workforce
#=========================================
cd evaluation/terminal_bench_run/
bash run_agent.sh \
        -a <attempt,0..n> \
        -n <total_attempts> \
        -e <conda env name> \
        -w <use_workforce>  # can have a try, focus on single chat agent now.
```


#### Log folder explaination
```
└── play-zork
    └── play-zork.1-of-1.test_run       # trial name
        ├── CAMEL_WORKDIR               # not used at the moment
        ├── agent-logs                  # not used at the moment
        ├── commands.txt                # not used at the moment
        ├── chatagent.log               # ❗️❗️ full history of running agent including test results
        ├── eigent_logs.json            # ⚠️ exists only when running workforce
        ├── panes                       # not used at the moment
        └── sessions                    # session logs
            ├── agent.cast              # not used at the moment
            ├── agent.log               # not used at the moment
            ├── session_logs            # ❗️❗️session logs for terminal toolkit
            │   ├── blocking_commands.log                   # ❗️❗️all block mode commands + output
            │   ├── session_run_zork_1_correct_path.log     # ❗️❗️non-block mode single session command + output
            │   ├── session_zork-1.log                      # ❗️❗️same as above session_{id}.log
            │   └── session_zork_start.log                  # ❗️❗️same as above session_{id}.log
            ├── tests.cast              # not used at the moment    
            ├── tests.log               # ❗️❗️test log
            └── tests.log.strip         # ❗️❗️test log with ansi control characters removed
```
### Run terminal bench official evaluation
```bash
cd evaluation/terminal_bench_eval/

# terminal bench 1.0
bash run_eval.sh

# terminal bench 2.0
bash run_tb2.sh

## The agent class is implemented in tbench_camel_agent.py 
```
#### ❗️Note: Results of evaluation
    - final results will be in `evaluation/terminal_bench_eval/run/{run_id}/results.json`
    
    - task specific terminal session logs will be in `evaluation/terminal_bench_eval/logs/camel_logs/{task_id}/`

### Train terminal agent

Everything is under training folder

Please refer to `training/tbench_areal_workflow/README.md` for setting up.