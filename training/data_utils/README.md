# Convert a dataset to parquet training format


## Current pipeline to use a new dataset

1. Download dataset

```bash
python download_data.py <dataset_name>
```

Choose `dataset_name` among

- synth_data
- tbench_core
- tbench_test
- tbench_adapted

2. Convert dataset to a table (parquet file)

```bash
python convert_tasks_to_dataset.py --tasks_dir dataset/<dataset_name> --output_dir dataset/<dataset_name>_convert/
```

3. Evaluate/ Train the data with baseline model