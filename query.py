QUERY = """
select user_item.id
    , user_item.user_id
    , user_item.item_id
    , brand.id || '__' || item.model || '__' || gender.id || '__' ||  user_item.size as sku_id
    , item.asin    
    , brand.name as brand_name
    , item.model
    , gender.name as gender
    , cast(user_item.size as float) as size
--    , cast(user_item.size_in as float) as size_in
    , cast(user_item.rating as float) as rating
    , cast(user_item.fit as float) as fit
    , item.brand_id
    , item.type as item_type
from user_item
left join item
    on item.id = user_item.item_id
left join brand
    on item.brand_id = brand.id
left join gender
    on item.gender_id = gender.id
"""
