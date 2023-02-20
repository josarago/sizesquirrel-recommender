QUERY = """
select user_item.id
    , user_item.user_id
    , user_item.item_id
    , cast(user.street_shoe_size_in as float) as street_shoe_size_in
    , user.foot_shape as user_foot_shape
    , cast(user.gender as int) as user_gender
    , user.climbing_boulder
    , user.climbing_sport
    , user.climbing_trad
--    , item.asin    
    , brand.name as brand_name
    , item.model
    , gender.name as shoe_gender
    , user_item.size as size
--    , cast(user_item.size_in as float) as size_in
    , cast(user_item.rating as int) as rating
--    , cast(user_item.fit as float) as fit
--    , item.brand_id
    , item.type as item_type
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
