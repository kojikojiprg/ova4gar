# PERSON INDICATORS
POSITION_DEFAULT = {
    'ankle_th': 0.7,    # Ankleキーポイントに対する閾値
    'size': 50,
    'ratio': 2.0,
    'std_th': 1,        # 外れ値に対する分散の閾値
}

# GROUP INDICATORS
DENSITY_DEFAULT = {
    'k': 3,
}
ATTENTION_DEFAULT = {
    'angle': 10,        # 視野角
    'division': 5,      # 計算するピクセル幅
}
PASSING_DEFAULT = {
    'th': 0.4,          # 確率のしきい値
    'th_shita': 60,     # cos類似度（向き合っている角度）のしきい値
    'gauss_mu': 120,    # 距離に対するガウス分布の平均
    'gauss_sig': 50,    # 距離に対するガウス分布の分散
}
