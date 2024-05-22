import _tkinter
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog


class Data:
    def __init__(self, master):
        super().__init__(master)
        self.main_instance = None
        self.master = master
        self.df_raw = None



# def optimize_pls_window(self):
#
#     pls_opt_window = PLSOptimize(self.master, self.df_X,
#                                  self.df_y)
#
#     self.master.wait_window(pls_opt_window.window)
#
#     self.df_X = pls_opt_window.df_X.copy()
#     self.df_y = pls_opt_window.df_y.copy()
#
#     self.display_table(self.features_data_tree, self.df_X)
#     self.display_table(self.labels_data_tree, self.df_y)
#
#     self.x_rows.set(str(self.df_X.shape[0]))
#     self.x_cols.set(str(self.df_X.shape[1]))
#     self.y_rows.set(str(self.df_y.shape[0]))
#     self.y_cols.set(str(self.df_y.shape[1]))
#
#     self.fig1.clear()
#
#     ax1 = self.fig1.add_subplot(111)
#
#     # Creating the legend table
#     legend_table = ax1.table(cellText=[[f'Train', f'{pls_opt_window.rmse_train:.6f}',
#                                         f'{pls_opt_window.r2_train:.6f}'],
#                                        [f'CV', f'{pls_opt_window.rmse_cv:.6f}',
#                                         f'{pls_opt_window.r2_cv:.6f}'],
#                                        [f'Test', f'{pls_opt_window.rmse_test:.6f}',
#                                         f'{pls_opt_window.r2_test:.6f}']],
#                              colLabels=['', 'RMSE', 'R-Square'],
#                              loc='upper left',
#                              cellLoc='center',
#                              cellColours=[['w', 'b', 'b'], ['w', 'r', 'r'], ['w', 'g', 'g']])
#
#     # Styling the legend table
#     legend_table.auto_set_font_size(False)
#     legend_table.set_fontsize(10)
#     legend_table.scale(0.4, 1.2)  # Adjust the size of the legend table
#
#     z = np.polyfit(pls_opt_window.y_train,
#                    pls_opt_window.y_pred_train, 1)
#
#     ax1.scatter(pls_opt_window.y_train, pls_opt_window.y_pred_train,
#                 color='blue', s=20, label="Train")
#     ax1.scatter(pls_opt_window.y_train, pls_opt_window.y_pred_cv,
#                 color='red', s=3, label="CV")
#     ax1.scatter(pls_opt_window.y_test, pls_opt_window.y_pred_test,
#                 color='green', s=3,
#                 label='Test')
#     ax1.plot(pls_opt_window.y_train, pls_opt_window.y_train,
#              color='k', linewidth=1, linestyle='--', label='Ideal Line')
#     ax1.plot(np.polyval(z, pls_opt_window.y_train), pls_opt_window.y_train,
#              color='b', linewidth=1, linestyle='--', label='Model Line')
#     ax1.set_title(f'Predicted vs True Results - Label={pls_opt_window.selected_label} |'
#                   f' PC#={pls_opt_window.nc} |'
#                   f' Test/Train={pls_opt_window.train_test_ratio * 100}%')
#     ax1.set_xlabel('True Values')
#     ax1.set_ylabel('Predicted Values')
#     ax1.legend(loc='lower right')
#     ax1.grid(True, alpha=0.3)
#     ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
#
#     self.fig2.clear()
#     ax2 = self.fig2.add_subplot(111)
#     ax2.plot(self.df_X.columns, pls_opt_window.pls.coef_[0, :])
#     ax2.set_xlabel('X columns')
#     ax2.set_ylabel('X Loadings')
#     ax2.set_title('PLS Weights')
#     # ax2.legend(loc='best')
#     ax2.grid(True, alpha=0.3)
#     ax2.xaxis.set_major_locator(plt.MaxNLocator(6))
#     #
#     self.fig3.clear()
#     # Find the index of the minimum RMSE score
#     min_rmse_index = np.argmin(pls_opt_window.rmse_scores)
#
#     ax3 = self.fig3.add_subplot(111)
#     ax3.plot(pls_opt_window.components_range, pls_opt_window.rmse_scores, marker='o')
#     ax3.plot(pls_opt_window.components_range[min_rmse_index], pls_opt_window.rmse_scores[min_rmse_index],
#              'P', ms=6, mfc='red',
#              label=f'Optimized Number of Components={pls_opt_window.components_range[min_rmse_index]}')
#     ax3.set_xlabel('Number of Components')
#     ax3.set_ylabel('RMSE')
#     ax3.set_title('RMSE vs Number of Components')
#     ax3.legend(loc='best')
#     ax3.grid(True)
#
#     self.fig4.clear()
#     ax4 = self.fig4.add_subplot(111)
#     ax4.plot(pls_opt_window.Tsq, pls_opt_window.Q, 'o')
#     ax4.plot([pls_opt_window.Tsq_conf, pls_opt_window.Tsq_conf], [ax4.axis()[2], ax4.axis()[3]], '--')
#     ax4.plot([ax4.axis()[0], ax4.axis()[1]], [pls_opt_window.Q_conf, pls_opt_window.Q_conf], '--')
#     ax4.set_title('Outliers Map')
#     ax4.set_xlabel("Hotelling's T-squared")
#     ax4.set_ylabel('Q residuals')
#
#     self.top_left_plot.draw()
#     self.top_right_plot.draw()
#     self.bottom_left_plot.draw()
#     self.bottom_right_plot.draw()
#
# def open_classification_window(self, class_type):
#     classification_window = Classification(self.master, self.df_X,
#                                            self.df_y, class_type)
#
#     self.master.wait_window(classification_window.window)
#
#     self.fig1.clear()
#     self.fig2.clear()
#     self.fig3.clear()
#     self.fig4.clear()
#
#     ax1 = self.fig1.add_subplot(111)
#     ax2 = self.fig2.add_subplot(111)
#     ax3 = self.fig3.add_subplot(111)
#     ax4 = self.fig4.add_subplot(111)
#
#     if classification_window.class_type == 'PCA':
#         scatter = ax1.scatter(-classification_window.x_pca[:, classification_window.x_pc],
#                               classification_window.x_pca[:, classification_window.y_pc],
#                               c=self.df_y[classification_window.selected_label].values)
#         ax1.set_title('PCA')
#         ax1.set_xlabel('PC' + str(classification_window.x_pc + 1))
#         ax1.set_ylabel('PC' + str(classification_window.y_pc + 1))
#         ax1.grid(True, alpha=0.3)
#         # Specify tick positions manually
#         ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
#
#         cbar = self.fig1.colorbar(scatter)
#         cbar.set_label(classification_window.selected_label)
#
#         ax2.plot(range(1, len(classification_window.explained_variance_ratio) + 1),
#                  classification_window.explained_variance_ratio, '-ob',
#                  label='Explained Variance ratio')
#         ax2.plot(range(1, len(classification_window.explained_variance_ratio) + 1),
#                  np.cumsum(classification_window.explained_variance_ratio), '-or',
#                  label='Cumulative Variance ratio')
#         ax2.set_title('Explained Variance')
#         ax2.set_xlabel('PC Number')
#         ax2.set_ylabel('Ratio')
#         ax2.grid(True, alpha=0.3)
#         ax2.legend(loc='best')
#         ax2.set_xlim(0.5, 10.5)
#         ax2.xaxis.set_major_locator(plt.MaxNLocator(6))
#
#         cmap = plt.get_cmap('viridis')
#         for i, j in enumerate(classification_window.eucl_dist):
#             arrow_col = (classification_window.eucl_dist[i] - np.array(classification_window.eucl_dist).min()) / (
#                     np.array(classification_window.eucl_dist).max() -
#                     np.array(classification_window.eucl_dist).min())
#             ax3.arrow(0, 0,  # Arrows start at the origin
#                       classification_window.ccircle[i][0],  # 0 for PC1
#                       classification_window.ccircle[i][1],  # 1 for PC2
#                       lw=2,  # line width
#                       length_includes_head=True,
#                       color=cmap(arrow_col),
#                       fc=cmap(arrow_col),
#                       head_width=0.05,
#                       head_length=0.05)
#             ax3.text((classification_window.ccircle[i][0]), (classification_window.ccircle[i][1]),
#                      self.df_X.columns[i], size=8)
#         # Draw the unit circle, for clarity
#         circle = Circle((0, 0), 1, facecolor='none', edgecolor='k', linewidth=1, alpha=0.5)
#         ax3.add_patch(circle)
#         ax3.set_title('PCA Correlation Circle')
#         ax3.set_xlabel('PC1')
#         ax3.set_ylabel('PC2')
#         ax3.set_xlim(-1, 1)
#         ax3.set_ylim(-1, 1)
#         ax3.axis('equal')
#         ax3.grid(True, alpha=0.3)
#         ax3.xaxis.set_major_locator(plt.MaxNLocator(6))
#
#         arr_2d = np.array(classification_window.eucl_dist).reshape(-1, 1)
#         scaler = MinMaxScaler()
#         scaler.fit(arr_2d)
#
#         eucl_dist_scaled = scaler.transform(arr_2d).flatten()
#         for i in range(self.df_X.shape[1]):
#             ax4.scatter(float(self.df_X.columns.values[i]), self.df_X.mean(axis=0).values[i],
#                         color=cmap(eucl_dist_scaled[i]))
#         ax4.set_title('Correlation Bands')
#         ax4.set_xlabel('X columns')
#         ax4.set_ylabel('Value')
#
#         ax4.xaxis.set_major_locator(plt.MaxNLocator(6))
#         ax4.grid(True, alpha=0.3)
#
#     elif classification_window.class_type == 'LDA':
#
#         cluster_means = []
#         for label in np.unique(classification_window.y_binned):
#             cluster_means.append(np.mean(classification_window.x_lda[classification_window.y_binned == label],
#                                          axis=0))
#
#         scatter = ax1.scatter(classification_window.x_lda[:, classification_window.x_pc],
#                               classification_window.x_lda[:, classification_window.y_pc],
#                               c=classification_window.y_binned, alpha=0.5)
#         ax1.scatter(np.array(cluster_means)[:, classification_window.x_pc],
#                     np.array(cluster_means)[:, classification_window.y_pc],
#                     c='red', marker='x', label='Cluster Means')
#         ax1.set_title('LDA')
#         ax1.set_xlabel('PC' + str(classification_window.x_pc + 1))
#         ax1.set_ylabel('PC' + str(classification_window.y_pc + 1))
#         ax1.legend(loc='best')
#         ax1.grid(True, alpha=0.3)
#         # Specify tick positions manually
#         ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
#
#         cbar = self.fig1.colorbar(scatter)
#         cbar.set_label(classification_window.selected_label + ' (binned)')
#
#         ax2.plot(range(1, len(classification_window.explained_variance_ratio) + 1),
#                  classification_window.explained_variance_ratio, '-ob',
#                  label='Explained Variance ratio')
#         ax2.plot(range(1, len(classification_window.explained_variance_ratio) + 1),
#                  np.cumsum(classification_window.explained_variance_ratio), '-or',
#                  label='Cumulative Variance ratio')
#         ax2.set_title('Explained Variance')
#         ax2.set_xlabel('PC Number')
#         ax2.set_ylabel('Ratio')
#         ax2.grid(True, alpha=0.3)
#         ax2.legend(loc='best')
#         ax2.set_xlim(0.5, 10.5)
#         ax2.xaxis.set_major_locator(plt.MaxNLocator(6))
#
#         cmap = plt.get_cmap('viridis')
#         for i, j in enumerate(classification_window.eucl_dist):
#             arrow_col = (classification_window.eucl_dist[i] - np.array(classification_window.eucl_dist).min()) / (
#                     np.array(classification_window.eucl_dist).max() -
#                     np.array(classification_window.eucl_dist).min())
#             ax3.arrow(0, 0,  # Arrows start at the origin
#                       classification_window.ccircle[i][0],  # 0 for PC1
#                       classification_window.ccircle[i][1],  # 1 for PC2
#                       lw=2,  # line width
#                       length_includes_head=True,
#                       color=cmap(arrow_col),
#                       fc=cmap(arrow_col),
#                       head_width=0.05,
#                       head_length=0.05)
#             ax3.text((classification_window.ccircle[i][0]), (classification_window.ccircle[i][1]),
#                      self.df_X.columns[i], size=8)
#         # Draw the unit circle, for clarity
#         circle = Circle((0, 0), 1, facecolor='none', edgecolor='k', linewidth=1, alpha=0.5)
#         ax3.add_patch(circle)
#         ax3.set_title('LDA Correlation Circle')
#         ax3.set_xlabel('PC1')
#         ax3.set_ylabel('PC2')
#         ax3.set_xlim(-1, 1)
#         ax3.set_ylim(-1, 1)
#         ax3.axis('equal')
#         ax3.grid(True, alpha=0.3)
#         ax3.xaxis.set_major_locator(plt.MaxNLocator(6))
#
#         arr_2d = np.array(classification_window.eucl_dist).reshape(-1, 1)
#         scaler = MinMaxScaler()
#         scaler.fit(arr_2d)
#
#         eucl_dist_scaled = scaler.transform(arr_2d).flatten()
#         for i in range(self.df_X.shape[1]):
#             ax4.scatter(float(self.df_X.columns.values[i]), self.df_X.mean(axis=0).values[i],
#                         color=cmap(eucl_dist_scaled[i]))
#         ax4.set_title('Correlation Bands')
#         ax4.set_xlabel('X columns')
#         ax4.set_ylabel('Value')
#
#         ax4.xaxis.set_major_locator(plt.MaxNLocator(6))
#         ax4.grid(True, alpha=0.3)
#
#     self.top_left_plot.draw()
#     self.top_right_plot.draw()
#     self.bottom_left_plot.draw()
#     self.bottom_right_plot.draw()
#
