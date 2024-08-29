import csv
import re
import random


def shuffle_frames(frames):
    tokens = re.findall(r"p\d+r\d+", frames)
    random.shuffle(tokens)
    return "".join(tokens)


def process_csv(input_file, output_file, n_new_rows=2, row_sample_rate=0.1):
    with open(input_file, "r", newline="") as infile, open(
        output_file, "w", newline=""
    ) as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            if random.random() > row_sample_rate:
                continue
            original_frames = row["frames"]
            for i in range(n_new_rows):  # Generate n new rows for each input row
                new_row = row.copy()
                new_row["name"] = f"{row['name']} (shuffled {i})"
                new_row["frames"] = shuffle_frames(original_frames)
                writer.writerow(new_row)
            # writer.writerow(row)  # Write the original row


if __name__ == "__main__":
    random.seed(42)
    input_file = "climbs.csv"
    output_file = "climbs_shuffled.csv"
    process_csv(input_file, output_file)
    print(f"Processed {input_file} and created {output_file} with shuffled frames.")
