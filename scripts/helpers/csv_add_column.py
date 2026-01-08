import csv

# ---------------------------
# User-defined settings
# ---------------------------
input_csv = "/home/gkini/Human-Traj-Prediction/scripts/data/synthetic/256_train/paths_5.csv"                  # Path to your input CSV
output_csv = input_csv 

new_column_name = "stations_pos/mask"
fill_value = 1

# ---------------------------
# Read original CSV
with open(input_csv, mode="r", newline="", encoding="utf-8") as f_in:
    reader = list(csv.DictReader(f_in))
    fieldnames = reader[0].keys()

# Add new column to fieldnames
fieldnames = list(fieldnames) + [new_column_name]

# Write new CSV with added column
with open(output_csv, mode="w", newline="", encoding="utf-8") as f_out:
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        row[new_column_name] = fill_value
        writer.writerow(row)

print(f"Done! New file saved to: {output_csv}")