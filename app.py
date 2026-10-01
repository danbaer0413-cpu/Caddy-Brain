with col_cont:
        st.subheader("🗺️ Contour Graph View")
        if contour_path and os.path.exists(contour_path):
            # Create figure with zero padding so the axes match the image frame exactly
            fig_cont, ax_cont = plt.subplots(figsize=(5, 6))
            fig_cont.subplots_adjust(left=0.12, right=0.98, bottom=0.08, top=0.95)
            
            c_img_obj = Image.open(contour_path).convert("RGB")
            max_depth_ft = saved_depth * 3.0
            green_width_ft = saved_width * 3.0
            
            # Display contour image as background matching data limits
            ax_cont.imshow(c_img_obj, extent=[0, green_width_ft, 0, max_depth_ft], origin='upper', aspect='auto')
            
            ax_cont.set_xticks(np.arange(0, green_width_ft + 1, 4))
            ax_cont.set_yticks(np.arange(0, max_depth_ft + 1, 5))
            ax_cont.grid(True, linestyle="--", alpha=0.6, color="#1f77b4")
            
            bx, by = st.session_state.ball_coords["x_ft"], st.session_state.ball_coords["y_ft"]
            hx, hy = st.session_state.hole_coords["x_ft"], st.session_state.hole_coords["y_ft"]
            
            aim_x = hx - aim_ft_val if aim_side == "Left" else hx + aim_ft_val
            ax_cont.plot([bx, aim_x], [by, hy], color="#1f77b4", linewidth=2, linestyle="--", label="Aim Line")
            
            t_vals = np.linspace(0, 1, 50)
            curve_dir = -1.0 if aim_side == "Left" else 1.0
            curve_x = (1 - t_vals)**2 * bx + 2 * (1 - t_vals) * t_vals * ((bx + hx)/2 + curve_dir * abs(slope_drop)*2.0) + t_vals**2 * hx
            curve_y = (1 - t_vals)**2 * by + 2 * (1 - t_vals) * t_vals * (by + hy)/2 + t_vals**2 * hy
            ax_cont.plot(curve_x, curve_y, color="#2e7d32", linewidth=3.5, label="True Curve")
            
            ax_cont.scatter([bx], [by], color="blue", s=80, zorder=5, edgecolors="white", label="Ball")
            ax_cont.scatter([hx], [hy], color="red", s=80, zorder=5, edgecolors="white", label="Hole")
            
            ax_cont.set_xlim(0, green_width_ft)
            ax_cont.set_ylim(0, max_depth_ft)
            ax_cont.set_facecolor("#fafafa")
            fig_cont.patch.set_facecolor("white")
            
            # Save without extra bounding box whitespace to keep pixel mapping true
            buf = BytesIO()
            fig_cont.savefig(buf, format="png", dpi=150)
            plt.close(fig_cont)
            buf.seek(0)
            cont_plot_img = Image.open(buf)
            
            clicked_cont_plot = streamlit_image_coordinates(cont_plot_img, key="contour_graph_click", width=380)
            if clicked_cont_plot is not None:
                # Precise transform mapping click pixels directly to data limits [0, width] and [0, height]
                # Accounting for the exact subplot margins used above (left=0.12, right=0.98, top=0.95, bottom=0.08)
                plot_w, plot_h = cont_plot_img.size
                
                # Bounding box offsets in pixel space from fig_cont layout
                px_left = plot_w * 0.12
                px_right = plot_w * 0.98
                px_bottom = plot_h * (1.0 - 0.08)
                px_top = plot_h * (1.0 - 0.95)
                
                ax_pixel_width = px_right - px_left
                ax_pixel_height = px_bottom - px_top
                
                click_px_x = clicked_cont_plot["x"]
                click_px_y = clicked_cont_plot["y"]
                
                # Linear interpolation mapping pixels to feet coordinates
                click_x_ft = ((click_px_x - px_left) / ax_pixel_width) * green_width_ft
                # Invert Y axis because image coordinate system starts from top-left (0,0) while Matplotlib is bottom-left
                click_y_ft = ((px_bottom - click_px_y) / ax_pixel_height) * max_depth_ft
                
                cx = max(0.0, min(click_x_ft, green_width_ft))
                cy = max(0.0, min(click_y_ft, max_depth_ft))
                
                if "Hole" in placement_mode:
                    st.session_state.hole_coords = {"x_ft": cx, "y_ft": cy}
                else:
                    st.session_state.ball_coords = {"x_ft": cx, "y_ft": cy}
                st.rerun()
        else:
            st.info("Contour map asset not found.")