"""Test script for subpanels 1, 2, 2.5, and 3 of figure 1."""

import matplotlib.pyplot as plt
from src import figure1

# Test subpanel 1 with Row1 configuration
print("Testing subpanel 1 with Row1_dict...")
fig1, ax1 = figure1.subpanel_1(figure1.Row1_dict)
plt.tight_layout()
plt.show()

print("Subpanel 1 test complete!")

# Test subpanel 2 with Row1 configuration
print("Testing subpanel 2 with Row1_dict...")
fig2, ax2 = figure1.subpanel_2(figure1.Row1_dict)
plt.tight_layout()
plt.show()

print("Subpanel 2 test complete!")

# Test subpanel 2.5 with Row1 configuration
print("Testing subpanel 2.5 with Row1_dict...")
fig25, ax25 = figure1.subpanel_2_5(figure1.Row1_dict)
plt.tight_layout()
plt.show()

print("Subpanel 2.5 test complete!")

# Test subpanel 3 with Row1 configuration
print("Testing subpanel 3 with Row1_dict...")
fig3, axes3 = figure1.subpanel_3(figure1.Row1_dict)
plt.show()

print("Subpanel 3 test complete!")

# Test subpanel 4 with Row1 configuration
print("Testing subpanel 4 with Row1_dict...")
try:
    fig4, ax4 = figure1.subpanel_4(figure1.Row1_dict)
    plt.tight_layout()
    plt.show()
    print("Subpanel 4 test complete!")
except Exception as e:
    print(f"Subpanel 4 test failed: {e}")
    import traceback
    traceback.print_exc()

