df = read.csv('/Users/akathian/Desktop/School/akathian05@gmail.com/LinuxDesk/word-recognition/data/training_data/0seed_100hid_0TestRuns_3000UpdateEpoch_0decay/analysis/181epoch/words_analysis_181.csv')
# str(df)
df$dominance <- as.factor(df$dominance)
df$frequency <- as.factor(df$frequency)
df$richness <- as.numeric(df$richness)

df$rt_thresh_stress0.7  <- as.numeric(df$rt_thresh_stress0.7)
df$rt_avg_st0.01 <- as.numeric(df$rt_avg_st0.01)
df$stress_at_tick_20 <- as.numeric(df$stress_at_tick_20)
df$avg_out_tick_20 <- as.numeric(df$avg_out_tick_20)
df$avg_out_diff_tick_20 <- as.numeric(df$avg_out_diff_tick_20)
# str(df)




rt = df$rt_thresh_stress0.7
# rt = df$rt_st0.01
# rt = df$stress_at_tick_20
# rt = df$avg_out_tick_20
# rt = df$avg_out_diff_tick_20
freq = df$frequency
numFeatures = df$richness
catDispersion = df$dominance

mod = lm(rt~freq*numFeatures*catDispersion)
summary(mod)
# hist(numFeatures)
print(summary(df))
hist(numFeatures)
# cor.test(freq,numFeatures)
# cor.test(freq,catDispersion)
# cor.test(rt,freq)


