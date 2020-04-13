'use strict';
const { colors, List, ListItem, Button, Typography, Slider } = MaterialUI;

const sliders = 4;

class Anat0Mixer extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            sliderValues: [0, 0, 0, 0],
            histogram: '|||',
            imageUrl: './imageUpdate?newValue=0,0,0,0',
        };
    }

    handleSlider(ndx, value) {
        var sliderValues = Object.assign({}, this.state.sliderValues);
        sliderValues[ndx] = value;
        this.setState({ sliderValues: sliderValues });

        var values = [sliderValues[0], sliderValues[1], sliderValues[2], sliderValues[3]].join();
        fetch('./imageUpdate?newValue=' + values)
            .then(response => response.blob())
            .then(responseAsBlob => URL.createObjectURL(responseAsBlob))
            .then(responseImageUrl => this.setState({ imageUrl: responseImageUrl }));

        fetch('./histogram?newValue=' + values)
            .then(response => response.text())
            .then(data => { this.setState({ histogram: data }) });
    };

    render() {
        return (
            <React.Fragment>
                <Typography id="vertical-slider" gutterBottom>
                    anat 0 mixer
                </Typography>
                <div style={{ height: 300 + 'px' }}>
                    {[0, 1, 2, 3].map(i =>
                        <Slider key={i}
                            orientation="vertical"
                            defaultValue={30}
                            aria-labelledby="vertical-slider"
                            onChange={(_, value) => this.handleSlider(i, value)}
                        />)}
                </div>
                <img alt="blank" src={this.state.imageUrl} style={{margin:30 + 'px'}} />
                <Typography>{'histogram: ' + this.state.histogram}</Typography>
            </React.Fragment>);
    }
}

var contentElement = document.getElementById('content');
ReactDOM.render(
    <Anat0Mixer />,
    contentElement
);
