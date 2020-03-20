'use strict';
const { colors, List, ListItem, Button, Typography, Slider } = MaterialUI;

const handleChange = (event, newValue) => {
    // setValue(newValue);
    fetch('http://127.0.0.1:5000/sliderValue?newValue=' + newValue)
        .then(response => response.json())
        .then(data => console.debug('data = ' + data));
};

const sliders = 4;

class Anat0Mixer extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            sliderValues: [0, 0, 0, 0],
            selectedObjectID: null,
            image: '',
        };
    }

    text() {
        var str = '';
        for (var n = 0; n < 4; n++) {
            str = str + this.state.sliderValues[n].toString();
        }
        return str;
    }

    showImage(responseAsBlob) {
        // Assuming the DOM has a div with id 'container'
        // var container = document.getElementById('showImage');
        // var imgElem = document.createElement('img');
        // container.appendChild(imgElem);
        var imgElem = document.getElementById('showImage');
        var imgUrl = URL.createObjectURL(responseAsBlob);
        imgElem.src = imgUrl;
    }

    componentDidMount() {
        // fetch('https://upload.wikimedia.org/wikipedia/commons/3/3f/KITTEN_on_BAMBOO_top_C_11JUN94.jpg')
        //     // .then(validateResponse)
        //     .then(response => response.blob())
        //     .then(this.showImage);
            // .catch(logError);
    }

    handleSlider(evt, ndx, value) {
        var sliderValues = Object.assign({}, this.state.sliderValues);
        sliderValues[ndx] = value;
        var values = [sliderValues[0], sliderValues[1], sliderValues[2], sliderValues[3]].join();
        this.setState({ sliderValues: sliderValues });
        fetch('http://127.0.0.1:5000/sliderValue?newValue=' + values)
            .then(response => response.text())
            .then(data => { this.setState({ image: data }); console.debug('response = ' + data) });

        fetch('http://127.0.0.1:5000/imageUpdate?newValue=' + values)
            .then(response => response.blob())
            .then(this.showImage);
    };

    render() {
        return (
            <React.Fragment>
                <Typography id="vertical-slider" gutterBottom>
                    anat0mixer
            </Typography>
                <div style={{ height: 300 + 'px' }}>
                    {[0, 1, 2, 3].map(i =>
                        <Slider key={i}
                            orientation="vertical"
                            defaultValue={30}
                            aria-labelledby="vertical-slider"
                            onChange={(evt, value) => this.handleSlider(evt, i, value)}
                        />)}
                </div>
                <Typography>{'blah: ' + this.state.image}</Typography>
            </React.Fragment>);
    }
}

var contentElement = document.getElementById('content');
ReactDOM.render(
    <Anat0Mixer />,
    contentElement
);