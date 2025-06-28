#!/usr/bin/env python3

import sqlite3
import csv
import os
import re
from pathlib import Path


def get_climbs_csv():
    """Extract climbs data from SQLite database and save as CSV."""

    # Paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / "data" / "climbs.sqlite3"
    csv_path = project_root / "data" / "climbs.csv"

    # Ensure data directory exists
    csv_path.parent.mkdir(exist_ok=True)

    # SQL query from the original script
    query = """
    SELECT
        climbs.`name`,
        climbs.`setter_username`,
        climbs.`description`,
        climbs.`angle`,
        climbs.`frames`,
        `climb_cache_fields`.`display_difficulty`,
        `climb_cache_fields`.`quality_average`
    FROM climbs
    INNER JOIN `climb_cache_fields` ON `climb_uuid` = `climbs`.`uuid`
    INNER JOIN product_sizes ON product_sizes.id = 10 /* 12 x 12 with kickboard */
    WHERE
        climbs.layout_id = 1 AND
        climbs.frames_count = 1 AND
        climbs.is_listed = 1 AND
        climbs.edge_left > product_sizes.edge_left AND
        climbs.edge_right < product_sizes.edge_right AND
        climbs.edge_bottom > product_sizes.edge_bottom AND
        climbs.edge_top < product_sizes.edge_top AND
        `climb_cache_fields`.`display_difficulty` IS NOT NULL AND
        `climb_cache_fields`.`quality_average` IS NOT NULL AND
        frames NOT REGEXP 'p139[6-9]' AND
        frames NOT REGEXP 'p14[0-3][0-9]' AND
        frames NOT REGEXP 'p144[0-6]' AND
        frames NOT REGEXP 'r[23]' AND
        LENGTH(frames) >= 48 AND
        LENGTH(frames) <= 256;
    """

    def regexp(pattern, text):
        """Custom REGEXP function for SQLite."""
        if text is None:
            return False
        return bool(re.search(pattern, str(text)))

    try:
        # Connect to SQLite database
        with sqlite3.connect(db_path) as conn:
            # Register custom REGEXP function
            conn.create_function("REGEXP", 2, regexp)
            cursor = conn.cursor()
            cursor.execute(query)

            # Get column names
            columns = [description[0] for description in cursor.description]

            # Write to CSV
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(columns)  # Header
                writer.writerows(cursor.fetchall())  # Data

        print(f"Successfully exported climbs data to {csv_path}")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    get_climbs_csv()
