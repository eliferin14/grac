# Capture time
python3 timestamps_analisys_v2.py -f timestamp_dataset.csv -th 0.01 -g capture -b 0.25 --title "Frame capture" --skip_plot

# Landmarks with two hands
python3 timestamps_analisys_v2.py -f timestamp_dataset.csv -th 0.01 -g landmarks -b 2.5 --title "Palm and landmarks detection - 2 hands" -n 2 --skip_plot

# Gesture classification
python3 timestamps_analisys_v2.py -f timestamp_dataset.csv -th 0.01 -g gestures -b 0.25 --title "Gesture classification - 2 hands" -n 2 --skip_plot

# UI
python3 timestamps_analisys_v2.py -f timestamp_dataset.csv -th 0.01 -g drawing -b 0.25 --title "UI" -cm "Joint control" "Cartesian control (base)" "Cartesian control (end effector)" -n 2 -lhg L --skip_plot

# Jacobian
python3 timestamps_analisys_v2.py -f timestamp_dataset.csv -th 0.01 -g jacobian -b 0.25 --title "Jacobian calculation" --skip_plot

# IK
python3 timestamps_analisys_v2.py -f timestamp_dataset.csv -th 0.001 -g ik -b 1 --title "Call to inverse kinematics service" --skip_plot 

# Interpretations
python3 timestamps_analisys_v2.py -f timestamp_dataset.csv -th 0.01 -g interpret -b 1 --title "Gesture interpretation - Joint CM" -rhg one two -lhg fist one two three four palm -cm "Joint control" --skip_plot
python3 timestamps_analisys_v2.py -f timestamp_dataset.csv -th 0.01 -g interpret -b 2 --title "Gesture interpretation - Cartesian CM" -rhg one two -lhg fist one two three four palm -cm "Cartesian control (base)" "Cartesian control (end effector)" --skip_plot
python3 timestamps_analisys_v2.py -f timestamp_dataset.csv -th 0.01 -g interpret -b 2 --title "Gesture interpretation - Hand Mimic CM" -rhg pick -lhg one two three four -cm "Hand Mimic" --skip_plot

# Total 
python3 timestamps_analisys_v2.py -f timestamp_dataset.csv -th 0.01 -g total -b 2 --title "Execution time - Joint CM" -rhg one two -lhg fist one two three four palm -cm "Joint control" --skip_plot
python3 timestamps_analisys_v2.py -f timestamp_dataset.csv -th 0.01 -g total -b 2 --title "Execution time - Cartesian CM" -rhg one two -lhg fist one two three four palm -cm "Cartesian control (base)" "Cartesian control (end effector)" --skip_plot
python3 timestamps_analisys_v2.py -f timestamp_dataset.csv -th 0.01 -g total -b 2 --title "Execution time - Hand Mimic CM" -rhg pick -lhg one two three four -cm "Hand Mimic" --skip_plot