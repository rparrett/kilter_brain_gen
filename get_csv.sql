.mode csv
.header on
.output climbs.csv

SELECT
    `name`, `setter_username`, `description`, `frames`,
    `climb_cache_fields`.`display_difficulty`,
    `climb_cache_fields`.`quality_average`
FROM climbs
INNER JOIN `climb_cache_fields` ON `climb_uuid` = `climbs`.`uuid`
WHERE
    layout_id = 1 AND
    frames_count = 1 AND
    `climb_cache_fields`.`display_difficulty` IS NOT NULL AND
    `climb_cache_fields`.`quality_average` IS NOT NULL AND
    frames NOT REGEXP 'p139[6-9]' AND
    frames NOT REGEXP 'p14[0-3][0-9]' AND
    frames NOT REGEXP 'p144[0-6]' AND
    frames NOT REGEXP 'r[23]' AND
    LENGTH(frames) >= 48 AND
    LENGTH(frames) <= 256;

