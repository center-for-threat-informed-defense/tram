/*---------------------------------------
* Functionalities for the Analytics page
*---------------------------------------*/

//Make calls to the REST API
function restRequest(type, data, callback, endpoint='/tram/api') {
    $.ajax({
       url: endpoint,
       type: type,
       contentType: 'application/json',
       data: JSON.stringify(data),
       success: function(data, status, options) {
           callback(data);
       },
       error: function (xhr, ajaxOptions, thrownError) {
           console.log(thrownError);
       }
    });
}

//Populate the chart with the data selected by the user
function fill_table(content){
    let table = $("#ttp_table");
    table.empty();
    table.append('<thead>' +
                    '<tr>' +
                        '<td>Name</td>' +
                        '<td>Code</td>' +
                        '<td>Confidence</td>' +
                        '<td>Occurrences</td>' +
                    '</tr>' +
                '</thead>')
    if(content){
        let obj = JSON.parse(content);
        for(let i = 0; i < obj.length; i++){
            let row = obj[i];
            table.append(
                '<tr>'+
                    '<td><p>'+row["name"]+'</p></td>'+
                    '<td><p>'+row["code"]+'</p></td>'+
                    '<td><p>'+row["confidence"]+'</p></td>'+
                    '<td><p>'+row["occurrences"]+'</p></td>'+
                '</tr>');
        }
    }
}

//Function to select which report information to display in the chart
function show_ttps(selectedValue){
    if(selectedValue === "all"){
        restRequest('GET', {}, fill_table, '/tram/api?action=all_ttps');
        $("#table-header").text("All Reports");
    }
    else if(selectedValue === "curr"){
        restRequest('GET', {}, fill_table, '/tram/api?action=curr_ttps');
        $("#table-header").text("Current Reports (To-Do and Review)");
    }
    else if(selectedValue === "past"){
        restRequest('GET', {}, fill_table, '/tram/api?action=past_ttps');
        $("#table-header").text("Completed Reports");
    }
}

function exportTTP(option){
    function downloadObjectAsJson(data){
        let dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(data, null, 2));
        let downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download",  "ttps.json");
        document.body.appendChild(downloadAnchorNode);
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
    }
    
    restRequest('POST', {'action':'exportTTP','ttp':option}, downloadObjectAsJson, '/tram/api');
}
