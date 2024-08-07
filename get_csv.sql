.mode csv
.header on
.output climbs.csv

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

