import matplotlib.pyplot as plt

# 读取日志文件
with open('/workspace2/zixin/wesep/examples/librimix/v1/exp/BSRNN_D2V/train100/pipeline/train1.log', 'r') as f:
    lines = f.readlines()

# 提取SI_SNR数据
iter_list = []
si_snr_list = []
for line in lines:
    if 'TRAIN' in line:
        line_parts = line.strip().split('|')
        iter_val = int(line_parts[3].strip())
        epoch = int(line_parts[2].strip())
        si_snr_val = float(line_parts[5].strip())
        iter_list.append(iter_val + 6900 * epoch)
        si_snr_list.append(si_snr_val)

# Create the plot
plt.figure(figsize=(10, 5))

# Plot SI-SNR variation
plt.plot(iter_list, si_snr_list, marker='o', label='SI-SNR')

# Add horizontal line at the minimum value
min_si_snr = min(si_snr_list)
plt.axhline(min_si_snr, color='red', linestyle='--', label='Min SI-SNR')

# Annotate the minimum value on the y-axis
plt.text(0, min_si_snr, '{:.2f}'.format(min_si_snr), va='center', ha='left', backgroundcolor='w')

# Add legend, title, and x, y labels
plt.legend(loc='upper right')
plt.title('SI-SNR Variation')
plt.xlabel('Iteration')
plt.ylabel('SI-SNR')

# Save the plot as a .png file
plt.savefig('si_snr_plot.png')

# Show the plot
plt.show()