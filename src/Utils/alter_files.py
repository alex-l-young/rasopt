# ====================================================================================================
# File containing parameter file alteration functions.
# Alex Young
# June 2021
# ====================================================================================================

# Library imports.
import re
import os
import h5py
from datetime import datetime

# ====================================================================================================
def get_start_date(bXX_fp):
    """
    Gets the simulation start date from the bXX file.
    :param bXX_fp: File path to bXX file.
    :return: Start date as a datetime object.
    """
    # Get the starting date time in the bXX file.
    with open(bXX_fp, 'r') as f:
        for line in f.readlines():
            if 'Start Date/Time' in line:
                dt = re.search('=\s(.*)', line)
                start_date_str = dt.group(1)
                start_dt = datetime.strptime(start_date_str, '%d%b%Y %H%M')
                break

    return start_dt


def edit_end_date_bXX(bXX_fp, end_dt_str):
    """
    Updates the ending datetime in the bXX file for the simulation.
    :param bXX_fp: Path to bXX file.
    :param end_dt_str: End datetime string formatted as "%d%b%Y %H%M"
    :return: None. Overwrites end datetime in bXX file.
    """
    # Edit the bXX file to update the ending datetime.
    with open(bXX_fp, 'r') as f:
        lines = []
        for line in f.readlines():
            if 'End Date/Time' in line:
                dt = re.sub('=\s(.*)', '= ' + end_dt_str, line)
                lines.append(dt)
                print(dt)
                continue
            lines.append(line)

    with open(bXX_fp, 'w') as f:
        f.writelines(lines)


def edit_start_date_bXX(bXX_fp, start_dt_str):
    """
    Updates the start datetime in the bXX file for the simulation.
    :param bXX_fp: Path to bXX file.
    :param start_dt_str: Start datetime string formatted as "%d%b%Y %H%M"
    :return: None. Overwrites end datetime in bXX file.
    """
    # Edit the bXX file to update the ending datetime.
    with open(bXX_fp, 'r') as f:
        lines = []
        for line in f.readlines():
            if 'Start Date/Time' in line:
                dt = re.sub('=\s(.*)', '= ' + start_dt_str, line)
                lines.append(dt)
                print(dt)
                continue
            lines.append(line)

    with open(bXX_fp, 'w') as f:
        f.writelines(lines)


def edit_start_date_pXX(pXX_fp, start_dt_str):
    """
    Updates the starting datetime in the pXX file for the simulation.
    :param pXX_fp: Path to pXX file. Not pXX.hdf.
    :param start_dt_str: Start datetime string formatted as "%d%b%Y,%H%M"
    :return: None. Overwrites starting datetime in pXX file.
    """
    with open(pXX_fp, 'r') as f:
        lines = []
        for line in f.readlines():
            if 'Simulation Date' in line:
                dt = re.sub('(?<=Date=).*(?=,\d{2}\w{3})', start_dt_str, line)
                lines.append(dt)
                print(dt)
                continue
            lines.append(line)

    with open(pXX_fp, 'w') as f:
        f.writelines(lines)


def edit_end_date_pXX(pXX_fp, end_dt_str):
    """
    Updates the ending datetime in the pXX file for the simulation.
    :param pXX_fp: Path to pXX file. Not pXX.hdf.
    :param end_dt_str: End datetime string formatted as "%d%b%Y,%H%M"
    :return: None. Overwrites end datetime in pXX file.
    """
    with open(pXX_fp, 'r') as f:
        lines = []
        for line in f.readlines():
            if 'Simulation Date' in line:
                dt = re.sub('(?<=,\d{4},).*', end_dt_str, line)
                lines.append(dt)
                print(dt)
                continue
            lines.append(line)

    with open(pXX_fp, 'w') as f:
        f.writelines(lines)


def edit_mapping_interval_bXX(bXX_fp, interval_mins):
    """
    Updates the ending datetime in the bXX file for the simulation.
    :param bXX_fp: Path to bXX file.
    :param interval_mins: Mapping interval in minutes.
    :return: None. Overwrites mapping interval in bXX file.
    """
    # Unit conversion if the interval is greater than one hour.
    unit = 'MIN'
    interval = interval_mins
    if interval_mins >= 60:
        unit = 'HOUR'
        interval = int(interval_mins / 60)

    # Edit the bXX file to update the ending datetime.
    with open(bXX_fp, 'r') as f:
        lines = []
        edit_line = 0
        for line in f.readlines():
            if edit_line == 1:
                subbed = re.sub('\w+$', f'{interval}{unit}', line)
                lines.append(subbed)
                print(subbed)
                edit_line = 0 # Reset the edit line flag.
                continue

            if 'Computation Level Output' in line:
                # Set the edit flag = 1 so that the next line will be edited.
                edit_line = 1

            lines.append(line)

    with open(bXX_fp, 'w') as f:
        f.writelines(lines)


def edit_mapping_interval(pXX_fp, interval_path, interval_mins):
    # Unit conversion if the interval is greater than one hour.
    unit = 'MIN'
    interval = interval_mins
    if interval_mins >= 60:
        unit = 'HOUR'
        interval = int(interval_mins / 60)

    # Create the interval string and encode as bytes.
    interval_str = f'{interval}{unit}'
    interval_bytes = str.encode(interval_str)
    print(interval_bytes)

    # Open the pXX file and write the new interval in.
    f = h5py.File(pXX_fp, 'r+')
    key = 'Base Output Interval'
    f[interval_path].attrs[key] = interval_bytes
    f.close()


def edit_mapping_interval_pXX(pXX_fp, interval_mins):
    """
    Updates the Mapping Interval in the pXX file for the simulation.
    :param pXX_fp: Path to pXX file. Not pXX.hdf
    :param interval_mins: Mapping interval in minutes.
    :return: None. Overwrites mapping interval in pXX file.
    """
    # Unit conversion if the interval is greater than one hour.
    unit = 'MIN'
    interval = interval_mins
    if interval_mins >= 60:
        unit = 'HOUR'
        interval = int(interval_mins / 60)

    # Edit the pXX file to update the ending datetime.
    with open(pXX_fp, 'r') as f:
        lines = []
        for line in f.readlines():
            if 'Mapping Interval' in line:
                subbed = re.sub('\w+$', f'{interval}{unit}', line)
                lines.append(subbed)
                print(subbed)
                continue

            lines.append(line)

    with open(pXX_fp, 'w') as f:
        f.writelines(lines)


def set_end_restart(bXX_fp):
    """
    Turns on functionality where restart file of final time step will be created.
    :param bXX_fp: Path to the bXX file.
    :return: None
    """
    # Edit the bXX file to update the ending datetime.
    with open(bXX_fp, 'r') as f:
        lines = []
        for line in f.readlines():
            if 'Write Restart File' in line:
                subbed = re.sub('\w$', 'T', line)
                lines.append(subbed)
                print('TURNED ON RESTART', subbed)
                continue

            lines.append(line)

    with open(bXX_fp, 'w') as f:
        f.writelines(lines)


def turn_on_restart_output(pXX_fp):
    """
        Turns on functionality where restart file of final time step will be created.
        :param pXX_fp: Path to the pXX file.
        :return: None
        """
    # Edit the pXX file to update the ending datetime.
    with open(pXX_fp, 'r') as f:
        lines = []
        for line in f.readlines():
            if 'Write IC File at Sim End' in line:
                subbed = re.sub('(?<=d=).*', '1', line)
                lines.append(subbed)
                print('TURNED ON RESTART', subbed)
                continue

            lines.append(line)

    with open(pXX_fp, 'w') as f:
        f.writelines(lines)


def turn_off_restart_output(pXX_fp):
    """
        Turns on functionality where restart file of final time step will be created.
        :param pXX_fp: Path to the pXX file.
        :return: None
        """
    # Edit the pXX file to update the ending datetime.
    with open(pXX_fp, 'r') as f:
        lines = []
        for line in f.readlines():
            if 'Write IC File at Sim End' in line:
                subbed = re.sub('(?<=d=).*', '0', line)
                lines.append(subbed)
                print('TURNED OFF RESTART', subbed)
                continue

            lines.append(line)

    with open(pXX_fp, 'w') as f:
        f.writelines(lines)


def add_restart_to_bxx(bXX_fp, restart_fname):
    """
    Adds the restart file to run from to the bXX file.
    :param bXX_fp: Path to bXX file.
    :param restart_fname: Name of the restart file.
    :return: None
    """
    with open(bXX_fp, 'r') as f:
        lines = []
        add_lines = False
        editing = False
        for line in f.readlines():
            if 'Initial Conditions (use restart file?)' in line:
                lines.append(line)
                add_lines = True
                editing = True
                continue

            if add_lines is True:
                # Set the initial condition flag to T.
                subbed = re.sub('\w$', 'T', line)
                lines.append(subbed)
                lines.append('Restart File\n')
                lines.append(f'{restart_fname}\n')
                add_lines = False
                continue

            if editing is True:
                # Skip any lines that need to be removed that may have been edited in the last loop.
                if 'Log File Information' in line:
                    lines.append(line)
                    editing = False
                    continue
                else:
                    continue

            lines.append(line)

    with open(bXX_fp, 'w') as f:
        f.writelines(lines)


def set_restart_uXX(uXX_fp, turn_on=True, restart_fname=None):
    """
    Turn on or off the use of a restart file in the uXX file.
    :param uXX_fp: Path to uXX file.
    :param turn_on: True if turning on the use of a restart. False if turning off.
    :param restart_fname: File name of restart file.
    :return: None
    """
    with open(uXX_fp, 'r') as f:
        lines = []
        editing = False
        for line in f.readlines():
            if 'Use Restart' in line:
                if turn_on is True:
                    lines.append('Use Restart=-1\n')
                    lines.append(f'Restart Filename={restart_fname}\n')
                    editing = True
                else:
                    lines.append('Use Restart=0\n')
                    editing = True
                    continue

            if 'Restart Filename' in line and turn_on is False:
                # Skip the restart filename line.
                continue

            if editing is True:
                if 'Boundary Location' in line:
                    lines.append(line)
                    editing = False
                    continue
                else:
                    continue

            # Add any other lines as before.
            lines.append(line)

    with open(uXX_fp, 'w') as f:
        f.writelines(lines)


def remove_restart_bXX(bXX_fp):
    """
    Removes any restart file and will set initial conditions to False.
    Does not mess with the specification of an output restart file, however.
    :param bXX_fp: Path to bXX file.
    :return: None
    """
    with open(bXX_fp, 'r') as f:
        lines = []
        add_lines = False
        editing = False
        for line in f.readlines():
            if 'Initial Conditions (use restart file?)' in line:
                lines.append(line)
                add_lines = True
                editing = True
                continue

            if add_lines is True:
                # Set the initial condition flag to T.
                subbed = re.sub('\w$', 'F', line)
                lines.append(subbed)
                add_lines = False
                continue

            if editing is True:
                # Skip any lines that need to be removed that may have been edited in the last loop.
                if 'Log File Information' in line:
                    lines.append(line)
                    editing = False
                    continue
                else:
                    continue

            lines.append(line)

    with open(bXX_fp, 'w') as f:
        f.writelines(lines)


def clear_restart_files(restart_dir):
    """
    Removes any restart files that are present in the directory.
    :param restart_dir: Directory to remove restart files from.
    :return: None
    """
    # Delete any .rst files at the end.
    dir_files = os.listdir(restart_dir)

    for file in dir_files:
        if file.endswith(".rst"):
            os.remove(os.path.join(restart_dir, file))


def update_input_hydrograph(uXX_fp, input_hydrograph):
    """
    Updates the input breach hydrograph in the model. Hydrograph is input in 10 item chunks in the file.
    :param uXX_fp: Path to unsteady flow input file.
    :param input_hydrograph: Input hydrograph as a numpy array.
    :return: None
    """
    # Generate list of strings from hydrograph rounded to 2 decimal places.
    num_str_list = ['{:.2f}'.format(i) for i in input_hydrograph]

    # Format lines of 10 numbers.
    template = [' ' for _ in range(8)]
    all_lines = []
    line = []
    c = 1
    for num_str in num_str_list:
        format_8 = template.copy()
        n_list = list(num_str)
        format_8[-len(n_list):] = n_list

        # Populate line.
        line.extend(format_8)

        if c == 10:
            line.append('\n')
            all_lines.append(line)
            line = []
            c = 0

        c += 1

    # Add the final incomplete line.
    line.append('\n')
    all_lines.append(line)

    # Write the new hydrograph data to the file.
    with open(uXX_fp, 'r') as f:
        editing = False
        skip_flow = True
        lines = []
        for line in f.readlines():
            if 'Flow Hydrograph=' in line and skip_flow is True:
                skip_flow = False
                lines.append(line)
                continue

            if 'Flow Hydrograph=' in line and skip_flow is False:
                lines.append(f'Flow Hydrograph= {len(num_str_list)}\n')
                for line in all_lines:
                    lines.append(''.join(line))
                editing = True
                continue
            elif editing is True and 'Stage Hydrograph TW Check' in line:
                editing = False
                lines.append(line)
                continue
            elif editing is True and 'Stage Hydrograph TW Check' not in line:
                continue

            lines.append(line)

    with open(uXX_fp, 'w') as f:
        f.writelines(lines)


def edit_final_bottom_width(pXX_fp, bottom_width):
    """
    Changes the final bottom width of the breach.
    :param pXX_fp: Path to .pXX file.
    :param bottom_width: Final bottom width of the breach in meters, must be integer value.
    :return: None
    """
    # Sub in the string.
    subbed = sub_breach_string(pXX_fp, 'Breach Geom=', str(bottom_width), 1)
    print('FINAL BOTTOM WIDTH LINE:', subbed)


def edit_weir_coef(pXX_fp, weir_coef):
    """
    Change the weir coefficient of the breach.
    :param pXX_fp: Path to .pXX file.
    :param weir_coef: Weir coefficient.
    :return: None
    """
    # Sub in the string.
    subbed = sub_breach_string(pXX_fp, 'Breach Geom=', str(weir_coef), -1)
    print('WEIR COEFFICIENT LINE:', subbed)


def edit_formation_time(pXX_fp, formation_time):
    """
    Edit breach formation time. This is the time to the full breach width.
    :param pXX_fp: Path to .pXX file.
    :param formation_time: Time to full formation of the breach.
    :return: None
    """
    # Sub in the string.
    subbed = sub_breach_string(pXX_fp, 'Breach Geom=', str(formation_time), -2)
    print('FORMATION TIME LINE:', subbed)


def sub_breach_string(fpath, line_match, sub_string, loc):
    """
    Sub a new value in at a location in the breach geometry string.
    :param fpath: Path to pXX file.
    :param line_match: String to identify the line that needs to be altered.
    :param sub_string: String to sub in.
    :param loc: Location of the string in the list, zero-indexed.
    :return: Subbed string.
    """
    # Read the lines from the file and edit.
    with open(fpath, 'r') as f:
        lines = []
        for line in f.readlines():
            if line_match in line:
                line_split = line.split(',')
                if loc == 0:
                    equal_split = line_split[loc].split('=')
                    equal_split[-1] = sub_string
                    sub_string = '='.join(equal_split)
                line_split[loc] = sub_string
                subbed = ','.join(line_split)
                lines.append(subbed+'\n')
            else:
                lines.append(line)

    # Write the edited lines back into the file.
    with open(fpath, 'w') as f:
        f.writelines(lines)

    return subbed



def sub_string(fpath, line_match, sub_string, reg_exp):
    """
    Searches file for the line match and subs the string
    in at the location specified by the regular expression.
    :param fpath: Path to file to edit.
    :param line_match: Part of line that is used to find the line to sub. Must be unique to line.
    :param sub_string: String to sub in.
    :param reg_exp: Regular expression.
    :return: The subbed line.
    """
    # Read the lines from the file and edit.
    with open(fpath, 'r') as f:
        lines = []
        for line in f.readlines():
            if line_match in line:
                subbed = re.sub(reg_exp, sub_string, line)
                lines.append(subbed)
            else:
                lines.append(subbed)

    # Write the edited lines back into the file.
    with open(fpath, 'w') as f:
        f.writelines(lines)

    return subbed


def edit_output_interval(pXX_fp, output_interval):
    """
    Edits the output interval in the 1D model.
    :param pXX_fp: Path to pXX file.
    :param output_interval: Output interval in minutes.
    :return: None.
    """
    output_unit = 'MIN'
    if output_interval >= 60:
        output_interval = int(output_interval / 60)
        output_unit = 'HOUR'
    with open(pXX_fp, 'r') as f:
        lines = []
        for line in f.readlines():
            if 'Output Interval' in line:
                line = f'Output Interval={output_interval}{output_unit}\n'
                print(line)

            if 'Instantaneous Interval' in line:
                line = f'Instantaneous Interval={output_interval}{output_unit}\n'
                print(line)

            lines.append(line)

    with open(pXX_fp, 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    import numpy as np
    uxx_fp = r'/Users/alexyoung/Desktop/Cornell/Research/Flood_Sim_DATA/Parameters/Secchia_Panaro.u01'
    Q = [12.1, 13.12]
    Q = np.array(Q)
    update_input_hydrograph(uxx_fp, Q)