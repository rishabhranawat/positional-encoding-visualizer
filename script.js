function generateHeatmap() {
    // Get input values
    const sentence = document.getElementById('sentence').value;

    // Prepare the request payload
    console.log(sentence);
    const data = { text: sentence };
    console.log(data);

    // Send a POST request to the Flask backend
    fetch('http://127.0.0.1:5000/encode', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(data => {
        // Call Plotly to generate a heatmap with the returned data
        const layout = {
            title: 'Positional Encoding Heatmap',
            xaxis: {
                title: 'Embedding Dimension',
            },
            yaxis: {
                title: 'Sequence Position',
            }
        };
        
        const tracePositionalEncodings = {
            z: data.positional_encodings,
            y: data.tokens, // Use tokens for labeling the y-axis
            type: 'heatmap',
            colorscale: 'Viridis',
            name: 'Positional Encodings'
        };

        // Trace for Combined
        const traceCombined = {
            z: data.combined,
            y: data.tokens, // Use tokens for labeling the y-axis
            type: 'heatmap',
            colorscale: 'Viridis',
            name: 'Combined Input + Positional Encoding'
        };

        Plotly.newPlot(
            'heatmapPositionalEncodings', 
            [tracePositionalEncodings], 
            Object.assign({}, layout, {title: 'Positional Encodings'})
        );

        Plotly.newPlot(
            'heatmapCombined', 
            [traceCombined], 
            Object.assign({}, layout, {title: 'Combined Input + Positional Encoding'})
        );
    })
    .catch(error => console.error('Error:', error));
}
