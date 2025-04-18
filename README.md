# Deep Reinforcement Learning (DRL) for military Course of Action (COA) anaylsis

This project implements a sophisticated reinforcement learning framework for training multi-agent systems in a custom military environment. Using a customized gymnasium environment - Proximal Policy Optimization (PPO) - and a progressive curriculum approach, agents learn increasingly complex behaviors - from basic navigation to advanced tactical operations involving terrain management and enemy engagement.

The system features a carefully designed reward structure that balances team and individual incentives, a discretized action space that reduces dimensionality while preserving tactical richness, and target validation functions that incorporate doctrinal principles. Results demonstrate that this approach produces emergent tactical behaviors closely resembling human military tactics without explicit programming.

This research contributes to the field of tactical AI by demonstrating how properly structured learning environments can produce sophisticated behaviors through reinforcement learning rather than rule-based systems, potentially improving both training simulations and autonomous tactical decision-making applications.

The following paper provides a detailed description of the project's scope and design (specifically the obervation space, action space, reward structure, and curriculum training protocol): ![Research Paper](https://github.com/ModSim-Steve/IDS_6916_Group-Project/blob/main/Moore_Lucernoni_Yingling_AI_for_Tactical_Planning_Route_Generation_and_Adaptive_Behavior.pdf)

![Agents in action](https://github.com/ModSim-Steve/EEL_6812_Project/blob/main/Demos/Scen1Episode.gif)

## How to - Train Agents:

1. Download the following files and place them into your python IDE drive:
- Friendly Force Composition File ![US Army Infantry PLT](https://github.com/ModSim-Steve/EEL_6812_Project/blob/main/US_Army_PLT_Composition_v2.py)
- Enemy Force Composition File ![Russian Armed Forces Assault Detachment](https://github.com/ModSim-Steve/EEL_6812_Project/blob/main/Russian_AF_ASLT_DET_Cap_SQD.py)
- Military Environment File ![War Gaming Environment](https://github.com/ModSim-Steve/EEL_6812_Project/blob/main/WarGamingEnvironment_v14.py)
- MARL Algorithm File ![Proximal Policy Optimization](https://github.com/ModSim-Steve/EEL_6812_Project/blob/main/PPO_Training_v4.py)
- Training Maps (you can use other maps or additional maps to seek a more robust tactical policy) ![Training Map LvL 1](https://github.com/ModSim-Steve/EEL_6812_Project/blob/main/training_map_lvl_1.xlsx) & ![Training Map LvL2](https://github.com/ModSim-Steve/EEL_6812_Project/blob/main/training_map_lvl_2.xlsx)

2. Train your agents with the MARL Algorithm file (pay attention to the bottom commented lines of code to progress to differing levels - AND - ensure the drive paths are updated accordingly to your IDE and project name).  
## How to - Develop & Evaluate COAs:

1. Determine what map you would like to use for your specific scenario and convert it to the format used within the WarGaming Environment.  We used ![Test Map 1](https://github.com/ModSim-Steve/EEL_6812_Project/blob/main/test_map_scenario1.xlsx), ![Test Map 2](https://github.com/ModSim-Steve/EEL_6812_Project/blob/main/test_map_scenario2.xlsx), and ![Test Map 3](https://github.com/ModSim-Steve/EEL_6812_Project/blob/main/test_map_scenario3.xlsx) for our findings in the research paper mentioned above. 

- 1.a. You can use the simple ![excel file](https://github.com/ModSim-Steve/EEL_6812_Project/blob/main/map_design.xlsx) to sketch out your map.  Ensure to use the terrain type encodings wihtin the cell (the double letter encoding will easily convert in the environment and will be conditionally rendered in the excel file for verification of the map).  Here are the encodings used:

![TerrainEncodings](https://github.com/ModSim-Steve/EEL_6812_Project/blob/main/Images/TerrainEncodings_excelrules.jpg)

- 1.b. Use the python file to convert the map design to the environment's format (ensure to update the drive path for the file).  ![Map Converter](https://github.com/ModSim-Steve/EEL_6812_Project/blob/main/Excel_to_CSV_Map_Converter.py)

2. Download the following files and place them into your python IDE drive:
- Tactical Position Analyzer File ![Tactical Position](https://github.com/ModSim-Steve/EEL_6812_Project/blob/main/tactical_position_analyzer.py)
- Tactical Route Analyzer File ![Tactical Route](https://github.com/ModSim-Steve/EEL_6812_Project/blob/main/TacticalRouteAnalyzer.py)
- Mission Success Analysis File ![Test File](https://github.com/ModSim-Steve/EEL_6812_Project/blob/main/PPO_Testing.py)

3. Develop a COA based upon your desired map using the Tactical Route Analyzer.  Future updates will include an automated linkage between all of the various files - but until then, you will have to specify the map, the friendly composition and starting locations, the enemy composition and starting locations, and the objective location.  The main function at the bottom of the file should be clear on where these inputs are required.  The final positions for each unit's task (assault, support by fire, or reserve) will be used as the input for the next step.

4. Determine mission success using the Mission Success Analysis file.  Again - future iterations will include automated linkages between all files - but at this point you will have to manually specify the map, friendly composition and starting locations, friendly agents policies, the enemy composition and starting locations, and the objective location.
