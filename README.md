# Deep Reinforcement Learning (DRL) for military Course of Action (COA) anaylsis

This project implements a sophisticated reinforcement learning framework for training multi-agent systems in a custom military environment. Using a customized gymnasium environment - Proximal Policy Optimization (PPO) - and a progressive curriculum approach, agents learn increasingly complex behaviors - from basic navigation to advanced tactical operations involving terrain management and enemy engagement.

The system features a carefully designed reward structure that balances team and individual incentives, a discretized action space that reduces dimensionality while preserving tactical richness, and target validation functions that incorporate doctrinal principles. Results demonstrate that this approach produces emergent tactical behaviors closely resembling human military tactics without explicit programming.

This research contributes to the field of tactical AI by demonstrating how properly structured learning environments can produce sophisticated behaviors through reinforcement learning rather than rule-based systems, potentially improving both training simulations and autonomous tactical decision-making applications.

![Research Paper](link)

![Agents in action](https://github.com/ModSim-Steve/EEL_6812_Project/blob/main/Demos/Scen1Episode.gif)

## How to - Train Agents:

## How to - Develop & Evaluate COAs:

1. Determine what map you would like to use for your specific scenario and convert it to the format used within the WarGaming Environment.
1.a. You can use the simple ![excel file](https://github.com/ModSim-Steve/EEL_6812_Project/blob/main/map_design.xlsx) to sketch out your map.  Ensure to use the terrain type encodings wihtin the cell (the double letter encoding will easily convert in the environment and will be conditionally rendered in the excel file for verification of the map).  Here are the encodings used:
![TerrainEncodings](https://github.com/ModSim-Steve/EEL_6812_Project/blob/main/Images/TerrainEncodings_excelrules.jpg)
1.b. Use the python file to convert the map design to the environment's format (ensure to update the drive path for the file).  ![Map Converter](https://github.com/ModSim-Steve/EEL_6812_Project/blob/main/Excel_to_CSV_Map_Converter.py)
