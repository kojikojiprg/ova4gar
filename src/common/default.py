# PERSON INDICATORS
POSITION_DEFAULT = {
    'ankle_th': 0.5,    # Ankleキーポイントに対する閾値
    'size': 30,
    'ratio': 1.2,
    'std_th': 1,        # 外れ値に対する分散の閾値
}
FACE_DEFAULT = {
    'size': 20,
    'ratio': 2.7,       # 耳の位置 / 肩と腰の差分
    'std_th': 1,        # 外れ値に対する分散の閾値
}
BODY_DEFAULT = {
    'size': 20,
    'ratio': 2.2,       # 肩の位置 / 肩と腰の差分
    'std_th': 1,        # 外れ値に対する分散の閾値
}
ARM_DEFAULT = {
    'size': 5,
    'std_th': 1,        # 外れ値に対する分散の閾値
}

# GROUP INDICATORS
ATTENTION_DEFAULT = {
    # 'angle': 10,        # 視野角
    # 'division': 5,      # 計算するピクセル幅
    'angle_th': 15,
    'count_th': 2
}
PASSING_DEFAULT = {
    'gauss_mu': 100,    # 距離に対するガウス分布の平均
    'gauss_sig': 100,    # 距離に対するガウス分布の分散
    'C': 54,
    'gamma': 0.37
}
