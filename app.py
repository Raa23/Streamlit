class ChartBlock(BaseModel):
    type: Literal["chart"] = "chart"
    chart_type: Literal["line", "bar", "scatter", "pie", "histogram", "box", "heatmap", "3d_scatter", "plotly_json"]
    data: Dict[str, Any]  # Flexible data structure
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    plotly_config: Dict[str, Any] | None = None  # For raw Plotly figure JSON

class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    content: str

class FormulaBlock(BaseModel):
    type: Literal["formula"] = "formula"
    content: str
    display_mode: bool = True

class TableBlock(BaseModel):
    type: Literal["table"] = "table"
    headers: List[str]
    rows: List[List[str]]
    caption: str | None = None

class CodeBlock(BaseModel):
    type: Literal["code"] = "code"
    content: str
    language: str = "python"

class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    url: str
    caption: str | None = None

ContentBlock = Union[TextBlock, FormulaBlock, TableBlock, CodeBlock, ChartBlock, ImageBlock]

class StructuredResponse(BaseModel):
    blocks: List[ContentBlock] = Field(description="List of content blocks in order")


class StructuredContentRenderer:
    """Render structured content blocks with full Plotly support"""
    
    def render(self, response: StructuredResponse):
        for block in response.blocks:
            self._render_block(block)
    
    def _render_block(self, block: ContentBlock):
        if block.type == "text":
            self._render_text(block)
        elif block.type == "formula":
            self._render_formula(block)
        elif block.type == "table":
            self._render_table(block)
        elif block.type == "code":
            self._render_code(block)
        elif block.type == "chart":
            self._render_chart(block)
        elif block.type == "image":
            self._render_image(block)
    
    def _render_text(self, block: TextBlock):
        st.markdown(block.content)
    
    def _render_formula(self, block: FormulaBlock):
        if block.display_mode:
            st.latex(block.content)
        else:
            st.markdown(f"${block.content}$")
    
    def _render_table(self, block: TableBlock):
        df = pd.DataFrame(block.rows, columns=block.headers)
        if block.caption:
            st.caption(block.caption)
        st.dataframe(df, use_container_width=True)
    
    def _render_code(self, block: CodeBlock):
        st.code(block.content, language=block.language)
    
    def _render_chart(self, block: ChartBlock):
        """Render interactive Plotly charts"""
        
        if block.title:
            st.subheader(block.title)
        
        # Handle raw Plotly JSON (most flexible)
        if block.chart_type == "plotly_json":
            fig = go.Figure(block.plotly_config)
            st.plotly_chart(fig, use_container_width=True)
            return
        
        # Convert data to DataFrame
        df = pd.DataFrame(block.data)
        
        # Create Plotly figure based on chart type
        if block.chart_type == "line":
            fig = px.line(df, 
                         x=df.columns[0] if len(df.columns) > 0 else None,
                         y=df.columns[1:] if len(df.columns) > 1 else None,
                         title=block.title,
                         labels={'x': block.x_label, 'y': block.y_label})
        
        elif block.chart_type == "bar":
            fig = px.bar(df,
                        x=df.columns[0] if len(df.columns) > 0 else None,
                        y=df.columns[1:] if len(df.columns) > 1 else None,
                        title=block.title,
                        labels={'x': block.x_label, 'y': block.y_label})
        
        elif block.chart_type == "scatter":
            fig = px.scatter(df,
                           x=df.columns[0] if len(df.columns) > 0 else None,
                           y=df.columns[1] if len(df.columns) > 1 else None,
                           title=block.title,
                           labels={'x': block.x_label, 'y': block.y_label})
            
        elif block.chart_type == "pie":
            fig = px.pie(df,
                        names=df.columns[0],
                        values=df.columns[1] if len(df.columns) > 1 else None,
                        title=block.title)
        
        elif block.chart_type == "histogram":
            fig = px.histogram(df,
                             x=df.columns[0],
                             title=block.title,
                             labels={'x': block.x_label})
        
        elif block.chart_type == "box":
            fig = px.box(df,
                        y=df.columns,
                        title=block.title)
        
        elif block.chart_type == "heatmap":
            fig = px.imshow(df,
                          title=block.title,
                          labels=dict(color=block.y_label or "Value"))
        
        elif block.chart_type == "3d_scatter":
            fig = px.scatter_3d(df,
                              x=df.columns[0] if len(df.columns) > 0 else None,
                              y=df.columns[1] if len(df.columns) > 1 else None,
                              z=df.columns[2] if len(df.columns) > 2 else None,
                              title=block.title)
        
        else:
            st.error(f"Unknown chart type: {block.chart_type}")
            return
        
        # Update layout if custom labels provided
        if block.x_label or block.y_label:
            fig.update_layout(
                xaxis_title=block.x_label,
                yaxis_title=block.y_label
            )
        
        # Render interactive chart
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_image(self, block: ImageBlock):
        st.image(block.url, caption=block.caption)
