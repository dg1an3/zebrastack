'use strict';
const { colors, List, ListItem, Button, Typography } = MaterialUI;

const API = 'https://hn.algolia.com/api/v1/search?hitsPerPage=3&query=';
const DEFAULT_QUERY = 'redux';
class FetchMe extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            hits: [],
            selectedObjectID: null,
        };
    }

    componentDidMount() {
        fetch(API + DEFAULT_QUERY)
            .then(response => response.json())
            .then(data => this.setState({ hits: data.hits }));
    }

    handleListItemClick(event, ht){
        this.setState({selectedObjectID: ht.objectID});
        console.debug(this.state);
        var detailsElement = document.getElementById('details');
        ReactDOM.render(
            <Typography variant="h2" component="h3">
                {ht.title}
            </Typography>,
            detailsElement
        );        
    };
    
    render() {
        var hits = this.state.hits;
        return (
            <div>
                <div>
                    <div>Hit count = {hits.length}</div>
                    <List component="nav" aria-label="main mailbox folders">
                        {hits.map(ht => 
                            <ListItem button 
                                key={ht.objectID} 
                                selected={this.state.selectedObjectID === ht.objectID} 
                                onClick={event => this.handleListItemClick(event, ht)}>
                                {ht.title}
                            </ListItem>)}
                    </List>
                </div>
            </div>
        );
    }
}

var contentElement = document.getElementById('content');
ReactDOM.render(
    <FetchMe />,
    contentElement
);