from vvla_lingbot.presets import resolve_recipe


def test_new_fp8_pool_composition_does_not_relabel_the_archived_cap1_recipe():
    previous=resolve_recipe('posttopk-down-fp8')
    pooled=resolve_recipe('cache32-posttopk-down-fp8')
    assert previous['options']['graph_cache_capacity']==1
    assert previous['resolved']==previous['composition']=='posttopk-down-fp8'
    assert pooled['posttopk'] and pooled['routed_down_fp8'] and not pooled['true_batch']
    assert pooled['options']['graph_cache_capacity']==32 and pooled['options']['sealed_guard']
    assert pooled['resolved']=='cache32-posttopk-down-fp8' and not pooled['customized']
    assert not pooled['full_weight_release_qualified']
    changed=resolve_recipe('cache32-posttopk-down-fp8',{'graph_cache_capacity':8})
    assert changed['customized'] and changed['composition'].startswith('cap8/')
