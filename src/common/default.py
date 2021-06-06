# PERSON INDICATORS
POSITION_DEFAULT = {
    'size': 20,
    'ratio': 1.2,
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
