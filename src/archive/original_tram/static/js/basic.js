/*-----------------
* Global variables
* ----------------*/
let attackTechniqueList = []; //Store techniques pulled from Mitre Attack
let sentObjectID = ""; //Store the currently user-highlighted sentence
var globalMovingReport;
/*----------------------------
* Setting up global variables
*----------------------------*/
//Pull in all techniques from Mitre Att&ck and store in a global array
function listAttackData(){
    function getSearch(searches){
        let names = [];
        searches.forEach(function(search){
            names.push(search.description);
        });
        let uniqueNames = new Set(names);  //Remove any accidental duplicates
        attackTechniqueList = Array.from(uniqueNames);
    }
    restRequest('GET', {}, getSearch, '/tram/api?action=search')
}
listAttackData();

/*-----------
* Login Page
* ----------*/
//Show or hide the user password when logging in to TRAM
function togglePassword(){
    let password = document.getElementById("password");
    if (password.type === "password"){
        password.type = "text";
    }
    else {
        password.type = "password";
    }
}

/*--------------------------------
* Basic functions of the home page
*--------------------------------*/
$(document).ready(function () {
    refresh();
});

setInterval(refresh, 10000);
//Refresh the home page to reflect changes to reports
function refresh(){
    function showReports(reports){
        let todo_div = $('#reports-todo');
        todo_div.empty();
        let review_div = $('#reports-review');
        review_div.empty();
        let completed_div = $('#reports-completed');
        completed_div.empty();
        reports.reverse().forEach(function(report) {
            let template = $('#report-template').clone();
            template.data('identifier', report.id);
            if(report.name) {
                template.find('#report-title').text(report.name);
            } else if(report.url) {
                template.find('#report-title').text(report.url);
            } else if(report.file) {
                template.find('#report-title').text(report.file);
            } else {
                template.find('#report-title').text(report.id);
            }
            template.find('#report-title').append('<div style="margin-top: 5px;" id="file_date">File Date: '+report.file_date+'</div>');
            template.find('#report-title').append('<div style="margin-top: 5px; color:yellow" id="assigned_user">Assigned User: '+report.assigned_user+'</div>')
            template.find('#report-title').append('<data id='+report.id+'></data>')
            template.data('status', report.status);
            template.data('name', report.name);
            template.data('user',report.assigned_user);
            template.data('url', report.url);
            template.data('exports', report.exports);
            template.data('file', report.file);
            template.data('sentences', report.sentences);
            template.data('matches', report.matches);
            template.data('indicators', report.indicators);

            template.show();
            box = template.find('#report-box')
            box.removeClass()
            box.addClass('report-box')
            if(report.status == 'TODO') {
                todo_div.append(template)
            } else if(report.status === 'REVIEW'){
                box.addClass('review-report');
                review_div.append(template);
            } else if(report.status === 'COMPLETED') {
                box.addClass('completed-report');
                completed_div.append(template);
            }
        });
    }
    restRequest('GET', {}, showReports, '/tram/api?action=reports')
}

// method to handle drag and drop, when user begins dragging
function onDragStart(event){
    event
        .dataTransfer
        .setData('text/plain', event.target.id);
    
    globalMovingReport = event.target;
    event
        .currentTarget
        .style
        .backgroundColor = 'white';
}

// method to handle dragOver card
function onDragOver(event){
    event.preventDefault();
}

// method to handle dropping card
function onDrop(event){
    const id = event
        .dataTransfer
        .getData('text');
    const draggableElement = document.getElementById(id);
    const dropzone = event.target;
    dropzone.appendChild(draggableElement);
    event
        .dataTransfer
        .clearData();
    console.log(globalMovingReport.getElementsByTagName("data")[0].id)
    let data = {
        "action":"reports","id":globalMovingReport.getElementsByTagName("data")[0].id,
        "status":event.target.closest('.reporting-column').id
    };
    restRequest('POST', data, refresh);

    location.reload();

}

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

/*-----------------------------------
* Functions of the home page Nav bar
*-----------------------------------*/
//POST the link or file and generate a report when the Submit button is clicked
function createReport() {
    let URL_BOX = $('#upload-url');
    let url = URL_BOX.val();
    restRequest('POST', {"action":"reports","url": url}, refresh);
    URL_BOX.val('');
    refresh();
}

function pullRSS() {
    let URL_BOX = $('#rssInput');
    let url = URL_BOX.val();
    restRequest('POST', {"action":"rss","url": url}, refresh);
    URL_BOX.val('');
    refresh();
}

function updateReport(){
    let tbl = $('#reportTable');
    let data = {
        "action":"reports","url":tbl.find('#url').text(),"name":tbl.find('#name').text()
        ,"id":tbl.find('#id').text(),"user":tbl.find('#user option:selected').text()
        ,"status":tbl.find('#status option:selected').text()
    };
    restRequest('POST', data, refresh);
}

function reassess(){
    let tbl = $('#reportTable');
    let data = {"action":"reassess", "id":tbl.find("#id").text()};
    restRequest('POST', data, refresh);
    location.reload();
}

function retrainBaseModel(){
   let tbl = $('#reportTable');
    let data = {"action":"retrain","model_name":"base model"};
    restRequest('POST', data, refresh);
}

//Retrain the Regex model
function retrainRegex(){
   let tbl = $('#reportTable');
    let data = {"action":"retrain", "model_name":"regex"};
    restRequest('POST', data, refresh);
}

/*-------------------------------------
* Functionality for uploading a report
*-------------------------------------*/
//When the Upload button is clicked, call the function to upload a report
$('#uploadReport').on('change', function (event){
    if(event.currentTarget) {
        let filename = event.currentTarget.files[0].name;
        if(filename){
            uploadReport();
        }
    }
});

//POST the report to the endpoint
function uploadReport() {
    let file = document.getElementById('uploadReport').files[0];
    let fd = new FormData();
    fd.append('file', file);
    $.ajax({
         type: 'POST',
         url: '/upload/report',
         beforeSend: function(xhr){xhr.setRequestHeader('Directory', 'data/reports/');},
         data: fd,
         processData: false,
         contentType: false
    }).done(function (){
        location.reload();
    })
}

/*--------------------------------------------
* Functionality in the home page dropdown menu
*---------------------------------------------*/
//Function to open and close dropdown menus when the menu top is clicked
$(document).ready(function(){
    $(".dropdown-top").on("click", function(){
        $(document).find("#feed-dropdown").slideToggle("fast");
    });
    $(".dropdown").on("click", function(e){
        $(document).find("#reportExports").slideToggle("fast");
    });
});

//Close the dropdown menu when the user makes a selection or clicks outside the menu
$(document).on("click", function(event){
    let $trigger = $(".dropdown-top");
    let rssLink = document.getElementById('rssInput');
    let exportDrop = document.getElementById('exportBtn')
    if($trigger !== event.target && !$trigger.has(event.target).length && rssLink !== event.target && exportDrop != event.target){
        $(".dropdown-content").slideUp("fast");
    }
});

//Take in an RSS feed and generate a series of reports for each post in the feed
function pullRSS() {
    let URL_BOX = $('#rssInput');
    let url = URL_BOX.val();
    restRequest('POST', {"action":"rss","url": url}, refresh);
    URL_BOX.val('');
    refresh();
}

//Export all reports from Mitre Att&ck matrices and download as a json file
//NOTE: In the current version of TRAM, this option has been removed from the home page
function pullAttackRefs() {
    function downloadAttackRefs(data){
        let dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(data, null, 2));
        let downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", "attack_matrices.json");
        document.body.appendChild(downloadAnchorNode);
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
    }

    let report_id = $('#reportTable').find('#id').text();
    restRequest('POST', {'action':'export','report_id':report_id, 'type':type}, downloadObject, '/tram/api');
}


function updateUser(currentOption){
    function userList(users){
        var user_text = '<tr>' +
                        '<td><b><p style="color:white">User</p></b></td>' +
                        '<td><select id="user" name="Users" onchange="updateReport()">';
        var current = false;
        for(var i = 0; i < users.length;i++){ 
            if(users[i] === currentOption){
                user_text += '<option value="' + users[i] + '" selected>' + users[i] + '</option>';
                current = true;
            }else{
                user_text += '<option value="' + users[i] + '">' + users[i] + '</option>';
            }
        }
        if(current != true){
            user_text += '<option value="" selected>Please select user</option>'
        }
        user_text += '</select>' + '</div>' + '</div></td>' + '</tr>' +
        '<tr><td><button onclick="reassess()">reassess report</button></td></tr>';
        $('#reportTable').append(
            user_text
        )
    }
    restRequest('GET', {}, userList, '/tram/api?action=users'); //r.parent().data();
}

/*-----------------------------------------------------------
* Functionalities related to modal reports generated by TRAM
 *-----------------------------------------------------------*/
//Open a new page to view/edit the full text of a generated report when Edit button is clicked
//The url will reflect the report's unique identifier
function edit_report(r) {
    let id = $(r.closest("li")).data('identifier');
    location.href = '/report/' + id;
}

//When the View button is clicked, open the report with the Overview tab shown by default
function openReport(r) {
    $('#reportTable').empty();
    //$('#fullReport').empty();
    parent = $(r.closest("li"))
    $('#reportTable').append(
        '<tr>' +
            '<td><b><p style="color:yellow">INFORMATIONAL ROWS:</p></b></td>' +
            '<td><p id="INFORMATIONAL ROWS:" contenteditable="false" spellcheck="false" style="color:yellow">INFORMATIONAL ROWS:</p></td>' +
        '</tr>'
    )
    addNonEditable('ID', parent.data('identifier'));
    addNonEditable('URL', parent.data('url'));
    addNonEditable('File', parent.data('file'));
    $('#reportTable').append(
        '<tr>' +
            '<td><b><p style="color:yellow">EDITABLE ROWS:</p></b></td>' +
            '<td><p id="EDITABLE ROWS:" contenteditable="false" spellcheck="false" style="color:yellow">EDITABLE ROWS:</p></td>' +
        '</tr>'
    )
    addReportRow('Name', parent.data('name'));
    var currentOption = parent.data('user');
    updateUser(currentOption);
    updateStatus(parent.data('status'));
    $('#reportMatches').find("tr:gt(0)").remove();
    $('#indicatorMatches').find("tr:gt(0)").remove();
    /*
    $("#fullReport").append(
        '<thead>'+
            '<tr>' +
                '<td>Full Text</td>'+
                '<td>Techniques Found</td>'+
            '</tr>'+
        '</thead>'
    )
    */
    let count = 1;
    parent.data('sentences').forEach(function(sentence) {
        if(!Array.isArray(sentence.matches) || !sentence.matches.length) {
            addFullReportRow(sentence, count, "transparent");
            count++;
        } else {
            addFullReportRow(sentence, count, "green");
            count++;
                    sentence.matches.forEach(function(match) {
            if(match.search.tag === 'attack') {
            let status = 'Accepted';
            if (!match.accepted) { status = 'Rejected'; }
            addMatchRow(parent.data('identifier'), match.id, match.model, match.search.name, match.search.code,
                match.search.description, sentence.text, match.confidence, status);
                }
            })
        }
    });

    parent.data('matches').forEach(function(match) {
        if(match.search.tag === 'ioc') {
            addIndicatorRow(match.search.code, match.search.name, match.search.description);
        }
    });

    $('#reportExports').empty();
    parent.data('exports').forEach(function(e) {
        $('#reportExports').append(
            `<li onclick="exportReport('${e}')"><a href="#" onclick="exportReport('${e}')">${e}</a></li><br>`
            //`<button onclick="exportReport('${e}')" style="width:10%;float:right;margin:4px;">${e}</button>`
        )
    });

    $('#report-modal').show();
}

// close modal if the user clicks in the gutter outside the modal content
$(document).click(function(e){
    if(e.target.id == 'report-modal'){
        document.getElementById('report-modal').style.display = 'none';
    }
})

/*----------------------------------------
* Switching between tabs in a modal report
*----------------------------------------*/
//Activate whichever tab is clicked on by the user
$(document).ready(function() {
	$('.tabs .tab-links a').on('click', function(e) {
		var currentAttrValue = $(this).attr('href');
		$('.tabs ' + currentAttrValue).show().siblings().hide();
		$(this).parent('li').addClass('active').siblings().removeClass('active');
		e.preventDefault();
	});
});

/*-----------------------------------------
* Functionality related to the Overview Tab
-------------------------------------------*/
//add a non-editable row that is just information
function addNonEditable(field, value){
    $('#reportTable').append(
        '<tr>' +
            '<td><b><p style="color:gray">'+field+'</p></b></td>' +
            '<td><p id="'+field+'" contenteditable="false" spellcheck="false" style="color:gray">'+value+'</p></td>' +
        '</tr>'
    )
}

//Create a row at the top of the Overview tab with info on the report (url, identifier, etc.)
function addReportRow(field, value){
    $('#reportTable').append(
        '<tr>' +
            '<td><b><p style="color:white">'+field+'</p></b></td>' +
            '<td><p id="'+field+'" contenteditable="true" spellcheck="false" style="color:white" oninput="updateReport()">'+value+'</p></td>' +
        '</tr>'
    )
}

//Populate the identified matches into the Overview
function addMatchRow(report_id, match_id, model, name, code, description, sentence, confidence, status){
    let select = '<select id="match-status-'+match_id+'" onchange="updateMatch(\''+report_id+'\', \''+match_id+'\')" style="margin:0; width: 8em"><option selected value=1>Accepted</option><option value=0>Rejected</option></select>';
    if(status=='Rejected') {
        select = '<select id="match-status-'+match_id+'" onchange="updateMatch(\''+report_id+'\', \''+match_id+'\')" style="margin:0; width: 8em"><option value=1>Accepted</option><option selected value=0>Rejected</option></select>';
    }

    let tid = code;
    let split_tid = tid.split('.');
    let address = "https://attack.mitre.org/techniques/";
    for (let x in split_tid){
        address += split_tid[x] + "/";
    }
    let confidence_3_decimal = parseFloat(confidence.toFixed(3));

    $('#reportMatches').append(
        '<tr>' +
            '<td><p style="color:white">'+model+'</p></td>' +
            '<td><p style="color:white">'+name+'</p></td>' +
            '<td><a style="color:white" href="'+address+'" target="_blank">'+code+'</a></td>' +
            '<td><p style="color:white">'+description+'</p></td>' +
            '<td style="word-break: break-word"><p style="color:white">'+sentence+'</p></td>' +
            '<td><p style="color: white">'+confidence_3_decimal+'</p></td>'+
            '<td>'+select+'</td>' +
        '</tr>'
    )
}

//POST the current report values to update any changes made by the user
function updateReport(){
    let tbl = $('#reportTable');
    let data = {
        "action":"reports","url":tbl.find('#url').text(),"name":tbl.find('#name').text()
        ,"id":tbl.find('#id').text(),"user":tbl.find('#user option:selected').text()
        ,"status":tbl.find('#status option:selected').text()
    };
    restRequest('POST', data, refresh);
}

//POST updated matches
function updateMatch(report_id, match_id) {
    let status = parseInt($('#match-status-'+match_id).val());
    restRequest('POST', {'action': 'match', 'report_id': report_id, 'match_id': match_id, 'accepted': status}, refresh);
}

//Reflect which user is assigned to a particular report
function updateUser(currentOption){
    function userList(users){
        var user_text = '<tr>' +
                        '<td><b><p style="color:white">User</p></b></td>' +
                        '<td><select id="user" name="Users" onchange="updateReport()">';
        var current = false;
        for(var i = 0; i < users.length;i++){
            if(users[i] === currentOption){
                user_text += '<option value="' + users[i] + '" selected>' + users[i] + '</option>';
                current = true;
            }else{
                user_text += '<option value="' + users[i] + '">' + users[i] + '</option>';
            }
        }
        if(current != true){
            user_text += '<option value="" selected>Please select user</option>'
        }
        user_text += '</select>' + '</div>' + '</div></td>' + '</tr>';
        $('#reportTable').append(
            user_text
        )
    }
    restRequest('GET', {}, userList, '/tram/api?action=users'); //r.parent().data();
}

//Show the status of a given report, e.g. under review, complete, etc.
function updateStatus(currentOption){
    var status_text = '<tr>' +
                    '<td><b><p style="color:white">Status</p></b></td>' +
                    '<td><select id="status" name="Statuses" onchange="updateReport()">' +
                    //'<option value="QUEUE">QUEUE</option>' +
                    '<option value="TODO">TODO</option>' +
                    '<option value="REVIEW">REVIEW</option>' +
                    '<option value="COMPLETED">COMPLETED</option>' +
                    '</select>' + '</div>' + '</div></td>' + '</tr>';
    status_text = status_text.replace(currentOption + '"', currentOption + '" selected')
    $('#reportTable').append(
        status_text
    )
}

//Add a row to the IOC section of the report Overview
function addIndicatorRow(code, indicator, value){
    $('#indicatorMatches').append(
        '<tr>' +
            '<td><p style="color:white">'+code+'</p></td>' +
            '<td><p style="color:white">'+indicator+'</p></td>' +
            '<td><p style="color:white;word-break: break-word">'+value+'</p></td>' +
        '</tr>'
    )
}

//Download a copy of a report to the user's local machine
function exportReport(type) {
    function downloadObject(data){
        if(data['output_type'] === 'json'){
            downloadObjectAsJson(data);
        }
        /*else if(data['output_type'] === 'xml'){
            downloadObjectAsXml(data);
        }else if(data['output_type'] === 'yaml'){
            downloadObjectAsYaml(data)
        }*/
    }
    function downloadObjectAsJson(data){
        let dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(data, null, 2));
        let downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", report_id + ".json");
        document.body.appendChild(downloadAnchorNode);
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
    }

    let report_id = $('#reportTable').find('#id').text();
    restRequest('POST', {'action':'export','report_id':report_id, 'type':type}, downloadObject, '/tram/api');
}

//Delete the current opened report
function deleteReport(r){
    var check = confirm("Are you sure you want to delete this report?");
    if(check){
        let id = $('#reportTable').find('#id').text();
        if (typeof r !== 'undefined') {
            id = $(r.closest("li")).data('identifier');
        }
        let data = {
            "action":"reports","id":id
        };
        restRequest('DELETE', data, refresh);
    }
}



/*--------------------------------------------
* Functionality related to the Full Report Tab
*--------------------------------------------*/
//Return a String containing all the techniques identified in a given sentence
function getTechniques(sentence) {
    let techniques = [];
    for (let i = 0; i < sentence.matches.length; i++){
        techniques.push(sentence.matches[i].search.description);
    }
    return techniques.toString();
}

//Populate a row in the Full Report tab
function addFullReportRow(sentence, count, background){
    const sentenceID = sentence.id;
    let techniques = getTechniques(sentence);
    let rowNumber = count.toString();
    let techName = "techFound".concat(rowNumber);
    $("#fullReport").append(
        '<tr>'+
            '<td style="word-break: break-word"><p class="fullText" id="'+sentenceID+'" style="color: white;background-color: '+background+'">'+sentence.text+'</p></td>'+
            '<td><p id="'+techName+'" style="color: white">'+techniques+'</p></td>'+
        '</tr>');

    let currNode = document.getElementById(sentenceID);
    currNode.onselectstart = storeNodeInfo;

    function storeNodeInfo(){
    sentObjectID= sentence.id;
    }
}

$('#uploadReport').on('change', function (event){
    if(event.currentTarget) {
        let filename = event.currentTarget.files[0].name;
        if(filename){
            uploadReport();
        }
    }
});

function uploadReport() {
    let file = document.getElementById('uploadReport').files[0];
    let fd = new FormData();
    fd.append('file', file);
    $.ajax({
         type: 'POST',
         url: '/upload/report',
         headers: {'Directory': 'data/reports/', 'file_date': file.lastModifiedDate},
         //beforeSend: function(xhr){xhr.setRequestHeader('Directory', 'data/reports/');},
         data: fd,
         processData: false,
         contentType: false
    }).done(function (){
        location.reload();
    })
}

$(document).ready(function() {
	$('.tabs .tab-links a').on('click', function(e) {
		var currentAttrValue = $(this).attr('href');
		$('.tabs ' + currentAttrValue).show().siblings().hide();
		$(this).parent('li').addClass('active').siblings().removeClass('active');
		e.preventDefault();
	});
});

function pullAttackRefs() {
    function downloadAttackRefs(data){
        let dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(data, null, 2));
        let downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", "attack_matrices.json");
        document.body.appendChild(downloadAnchorNode);
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
    }
    restRequest('POST', {'action':'attack'}, downloadAttackRefs, '/tram/api');
}

let thisFullReport = document.getElementById("fullReport"); //The contents of the Full Report tab

thisFullReport.addEventListener("mouseup", function doSomethingWithSelectedText() {
    let selectedInfo = getSelectedText();
    let selectedText = selectedInfo['text'];
    if (selectedText) {
        add_popup_menu(selectedInfo.htmlInfo);
    }
});

//Returns the user selection when they click and drag their mouse over some text
function getSelectedText() {
    let text = "";
    let htmlInfo = [];
    let selectionCheck = window.getSelection().toString();
    if(!selectionCheck) {
        return {text:""};
    }
    else if (window.getSelection().getRangeAt(0).startContainer.parentElement.className === "fullText" &&
        window.getSelection().getRangeAt(0).startContainer.parentElement.className !== "dropList") {
            let userSelection = window.getSelection();
            text = userSelection.toString();
            let rangeObject = userSelection.getRangeAt(0);
            let contents = rangeObject.cloneContents();
            if (contents.children.length === 0) {
                htmlInfo.push(document.getElementById(sentObjectID));
            } else {
                for (let i = 0; i < contents.children.length; i++) {
                    htmlInfo.push(contents.children[i].children[0].children[0]);
                }
            }
            return {text: text, htmlInfo: htmlInfo};
        }
}

//Creates a dropdown menu directly underneath the specified html element
function add_popup_menu(htmlNodes){
    let lastLine = htmlNodes[htmlNodes.length-1].id;
    let lineElem = document.getElementById(lastLine);
    let sentenceID = [];
    let menuID = "dropDown"+lastLine;
    for(let node in htmlNodes){
        sentenceID.push(htmlNodes[node].id);
    }
    $(lineElem).append('<select class="dropList" id="'+menuID+'" onchange="updateTechniques(\''+menuID+'\', \''+ sentenceID+'\')">' +
        '<option id="menuHeader" value="" selected hidden>Add Technique</option>' +
        '</select>');
    let dropList = document.getElementById(menuID);
    let choices = attackTechniqueList
    for(let i in choices){
        $(dropList).append('<option value="'+choices[i]+'">'+choices[i]+'</option>');
    }
    window.getSelection().removeAllRanges();
}

//POST API request to create a user-identified match
function updateTechniques(menuID, sentenceID){
    let dropMenu = document.getElementById(menuID);
    restRequest('POST', {'action':'addMissing', 'match_desc':dropMenu.value, 'orig_sentenceID':sentenceID}, refreshFullReport);
    refresh();
}

$(document).ready(function() {
    $(window).keydown(function(event){
        if(event.keyCode == 13) {
            event.preventDefault();
            document.getElementById('upload-submit').click();
        }
    });
});

//Refresh the Full Report tab to reflect any user changes
function refreshFullReport(report){
    $('#fullReport').empty();
    $("#fullReport").append(
        '<thead>'+
            '<tr>' +
                '<td>Full Text</td>'+
                '<td>Techniques Found</td>'+
            '</tr>'+
        '</thead>'
    )
    let count = 1;
    report.sentences.forEach(function(sentence) {
        if(!Array.isArray(sentence.matches) || !sentence.matches.length) {
            addFullReportRow(sentence, count, "transparent");
            count++;
        } else {
            addFullReportRow(sentence, count, "green");
            count++;
        }
    });
}

/*--------------------
* Logging out of TRAM
*--------------------*/
//Logout the current user and return to login screen
function logout(){
    restRequest('GET', {}, setTimeout(
        function(){
            location.reload()
        },250
    ), '/logout');
}