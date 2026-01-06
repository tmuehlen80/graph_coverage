# Graph Construction Visualization

This folder contains scripts to visualize the graph construction process at different stages.

## Files

- `create_graph_after_discovery.py` - Creates a graph showing all connections after relation discovery (before redundancy prevention)
- `create_graph_construction_plots.py` - Creates visualization plots at different construction stages
- `actor_graph_setting_1_50_50_10_20_20_4_4_4.json` - Graph construction settings

## Usage

### Step 1: Generate the graph after discovery

```bash
poetry run python article/plots/graph_construction/create_graph_after_discovery.py --scenario-id YOUR_SCENARIO_ID
```

This will create:
- `{scenario_id}_graph_after_discovery.pkl` - Pickle file with the graph

### Step 2: Generate visualization plots

```bash
poetry run python article/plots/graph_construction/create_graph_construction_plots.py --scenario-id YOUR_SCENARIO_ID
```

This will create:
- `{scenario_id}_after_discovery.png` - All connections after relation discovery
- `{scenario_id}_after_step1.png` - After step 1 (leading/following edges only)
- `{scenario_id}_after_step2.png` - After step 2 (leading + neighbor edges)
- `{scenario_id}_final.png` - Final graph (all steps with redundancy prevention)

## Command-line Options

Both scripts support the following options:

- `--scenario-id` - Scenario ID to process (default: `0922f82c-0640-43a6-b5ce-c42bc729418e`)
- `--timestamp` - Timestamp to use for the graph (default: `1.0`)
- `--dataroot` - Path to data root directory (default: `repo_root/argoverse_data/train`)

## Example

To process a different scenario:

```bash
# Step 1: Generate graph after discovery
poetry run python article/plots/graph_construction/create_graph_after_discovery.py --scenario-id a3a43e5a-4fbe-4e40-9447-21bc2dc666a7

# Step 2: Generate plots
poetry run python article/plots/graph_construction/create_graph_construction_plots.py --scenario-id a3a43e5a-4fbe-4e40-9447-21bc2dc666a7
```

## Notes

- You must run `create_graph_after_discovery.py` first before running `create_graph_construction_plots.py`
- The scripts use the graph settings from `actor_graph_setting_1_50_50_10_20_20_4_4_4.json`
- All output files are saved in the same directory as the scripts

