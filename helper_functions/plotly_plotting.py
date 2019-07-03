import plotly.plotly as py
import plotly.graph_objs as go

def plot_austin(latitudes, longitudes, mode='markers', marker_size=5, 
                text='Austin', height=1000, width=1000, hovermode='closest',
                austin_lat=30.2648, austin_lon=-97.7472, pitch=0, zoom=9.5):
    """Wrapper function for plotly map plot, with default function settings
    adjusted to map Austin, TX. Takes lists of latitudes and longitudes and 
    plots them in a plotly map.
    
    Arguments:
        latitudes {list} -- list of latitudes
        longitudes {list} -- list of longitudes
    
    Keyword Arguments:
        mode {str} --  (default: {'markers'})
        marker_size {int} --  (default: {5})
        text {str} --  (default: {'Austin'})
        height {int} --  (default: {1000})
        width {int} --  (default: {1000})
        hovermode {str} --  (default: {'closest'})
        austin_lat {float} --  (default: {30.2648})
        austin_lon {float} --  (default: {-97.7472})
        pitch {int} --  (default: {0})
        zoom {float} --  (default: {9.5})
    
    Returns:
        None
    """

    mapbox_access_token = 'pk.eyJ1IjoiYXVzdGlubWFkZXJ0IiwiYSI6ImNqdWVpOG1pcDAzdDg0M3BwajlvYzBvNGwifQ.QNvTS_HgoqwLXw9ZCBTUgA'

    data = [
        go.Scattermapbox(
            lat=latitudes,
            lon=longitudes,
            mode=mode,
            marker=go.scattermapbox.Marker(
                size=marker_size
            ),
            text=[text],
        )
    ]

    layout = go.Layout(
        #autosize=True,
        height=height,
        width=width,
        hovermode=hovermode,
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=austin_lat,
                lon=austin_lon
            ),
            pitch=pitch,
            zoom=zoom
        ),
    )

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='Austin Mapbox')