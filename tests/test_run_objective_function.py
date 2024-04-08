from growing_with_experience.blackbox_ppo import black_box_ppo

def test_black_box_ppo():
    result = black_box_ppo(None, None)
    assert result > 0
    assert isinstance(result, float)
    
if __name__ == '__main__':
    test_black_box_ppo()