# Python code for LA ICP MS imaging data segmentation

Thresholding based segmentation of LA-ICP-MS imaging data

# Description

[src/main.py](src/main.py) - run this to process LA ICP MS data in data folder - generates matplotlib.figures - project specific setup
[src/laicpms_data_handler.py](src/laicpms_data_handler.py) - contains object to import, handle and segment (shimadzu) raw data

# Dependencies

Python 3.8.1 or newer

For packages see [requirements.txt](requirements.txt)

# Data

LA-ICP-MS imaging data as plain text files (comma-separated values)

Condition 1 = fresh frozen (FF)
Condition 2 = room temperature vacuum dried and sealed (RTV)
Condition 3 = formalin fixed (FFix)
Condition 4 = formalin fixed, paraffin sealed (FFPS)

3 replicate sectioning sets named A, B, C

File-naming: LA_Data_CISN1.csv, where I = [1, 2, 3, 4] is indicating the condition used and N = [A, B, C] is indicating the replicate set

# License

## Data

CC-BY 4.0 - respective [LICENSE](data/LICENSE) file in data folder

## Source code

MIT - respective [LICENSE](src/LICENSE) file in src folder
