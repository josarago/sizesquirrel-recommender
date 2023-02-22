QUERY = """
select user_item.id
    , user_item.user_id
    , user_item.item_id
    , cast(user.street_shoe_size_in as float) as user_street_shoe_size_in
    , user.foot_shape as user_foot_shape
    , cast(user.gender as int) as user_gender
    , user.climbing_boulder as bouldering
    , user.climbing_sport as sport_climbing
    , user.climbing_trad as trad_climbing
--    , item.asin    
    , brand.name as brand_name
    , brand.id as brand_id
    , item.model
    , gender.name as shoe_gender
    , user_item.size as size
--    , cast(user_item.size_in as float) as size_in
--    , item.brand_id
    , item.type as item_type
--    , cast(user_item.fit as float) as fit
    , cast(user_item.rating as float) as rating
from user_item
left join item
    on item.id = user_item.item_id
left join brand
    on item.brand_id = brand.id
left join gender
    on item.gender_id = gender.id
left join user
    on user_item.user_id = user.id
"""
