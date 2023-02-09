select user_item.id
    , user_item.user_id
    , user_item.item_id
    , brand.id || '__' || item.model || '__' || gender.id || '__' ||  user_item.size as sku_id
    , item.asin    
    , brand.name as brand_name
    , item.model
    , gender.name as gender_name    
    , user_item.size
    , user_item.size_in
    , user_item.rating
    , user_item.fit
    , item.brand_id
    , item.type as item_type
    , ( item.count_beginner_climbers 
        + item.count_advanced_climbers 
        + item.count_expert_climbers
    ) as total_count_climbers
from user_item
left join item
    on item.id = user_item.item_id
left join brand
    on item.brand_id = brand.id
left join gender
    on item.gender_id = gender.id




select user_item.id
    , user_item.size_in
    , user_item.size
    , item.brand_id
    , item.gender_id
    , item.model
from user_item
inner join item
    on   user_item.item_id = item.id  
where user_item.id in (503, 1977)


select count(1) from user_item



select count(*) from user_item
