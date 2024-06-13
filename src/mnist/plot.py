import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Function to add a layer rectangle


def add_layer(ax, center, size, label, color='lightblue'):
    x, y = center
    width, height = size
    rect = patches.FancyBboxPatch((x - width/2, y - height/2), width, height,
                                  boxstyle="round,pad=0.1", linewidth=1.5, edgecolor='black', facecolor=color, alpha=0.7)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=10)

# Function to add boxes for encoder and decoder


def add_section(ax, center, size, label, color='lightgray'):
    x, y = center
    width, height = size
    rect = patches.FancyBboxPatch((x - width/2, y - height/2), width, height,
                                  boxstyle="round,pad=0.1", linewidth=1.5, edgecolor='black', facecolor=color, alpha=0.3)
    ax.add_patch(rect)
    ax.text(x, y + height/2 - 0.3, label, ha='center',
            va='center', fontsize=12, fontweight='bold')


# Add encoder and decoder sections
add_section(ax, (1, 4), (2.5, 10), 'Encoder', color='lightcoral')
add_section(ax, (9, 4), (2.5, 10), 'Decoder', color='lightgreen')

# Encoder layers
add_layer(ax, (1, 8), (2, 1), 'Input Image\n(Noise)')
add_layer(ax, (1, 6), (2, 1), 'Conv Layer + ReLU\n$H_1 = ReLU(W_1 X + b_1)$')
add_layer(ax, (1, 4), (2, 1), 'Downsample\n$H_2 = Down(H_1)$')
add_layer(ax, (1, 2), (2, 1), 'Conv Layer + ReLU\n$H_3 = ReLU(W_2 H_2 + b_2)$')
add_layer(ax, (1, 0), (2, 1), 'Downsample\n$H_4 = Down(H_3)$')

# Bottleneck
add_layer(ax, (5, 0), (2, 1), 'Bottleneck\nLatent Space\n$Z = Enc(X)$')

# Decoder layers
add_layer(ax, (9, 0), (2, 1), 'Upsample\n$H_5 = Up(Z)$')
add_layer(ax, (9, 2), (2, 1), 'Conv Layer + ReLU\n$H_6 = ReLU(W_3 H_5 + b_3)$')
add_layer(ax, (9, 4), (2, 1), 'Upsample\n$H_7 = Up(H_6)$')
add_layer(ax, (9, 6), (2, 1), 'Conv Layer + ReLU\n$H_8 = ReLU(W_4 H_7 + b_4)$')
add_layer(ax, (9, 8), (2, 1), 'Output Image\n(Denoised)\n$Y = Dec(Z)$')

# Skip connections
ax.annotate('', xy=(2.25, 6), xytext=(7.75, 6),
            arrowprops=dict(arrowstyle="-|>", color='gray', lw=1.5))
ax.annotate('', xy=(2.25, 2), xytext=(7.75, 2),
            arrowprops=dict(arrowstyle="-|>", color='gray', lw=1.5))

# Formatting
ax.set_xlim(-1, 11)
ax.set_ylim(-2, 10)
ax.axis('off')

# Display the diagram
plt.title('Stable Diffusion Model - UNet Architecture', fontsize=14)
plt.show()
