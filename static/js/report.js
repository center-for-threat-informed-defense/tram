let attackTechniqueList = []; //Store techniques pulled from Mitre Attack

/*----------------------------
* Setting up global variables
*----------------------------*/
//Pull in all techniques from Mitre Att&ck and store in a global array
function listAttackData(){
    function getSearch(searches){
        let names = [];
        searches.forEach(function(search){
            if(search.tag === 'attack'){
                names.push(search.description);
            }
        });
        let uniqueNames = new Set(names);  //Remove any accidental duplicates
        attackTechniqueList = Array.from(uniqueNames);
    }
    restRequest('GET', {}, getSearch, '/tram/api?action=search')
}
listAttackData();

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

function refresh(){
    console.log("REFRESHED!!!")
}

function populateDropdown(r){
    let element = r.next("div");
    if(element.children()[0].innerText.includes("DLL")){
        console.log("populated")
    }else{
        var innerHtml = element.children()[0].innerHTML;
        for(var j = 0; j < attackTechniqueList.length; j++){
            innerHtml += `<a onclick="addMatch($(this),'` + element.next()[0].id + `')">` + attackTechniqueList[j] + '</a>\n';
        }
        
        element.children()[0].innerHTML = innerHtml;
    }
}


function dropdown(r){
    populateDropdown(r)
    if(r.next("div").css("display") === "none"){
        r.next("div").css("display","block");
    }else{
        r.next("div").css("display","none");
    }
    
}

function filter(r){
    var filter, ul, li, a, i;
    console.log(r[0].value)
    filter = r[0].value.toUpperCase();
    div = r.parent().parent()[0];
    a = div.getElementsByTagName("a");
    for (i = 0; i < a.length; i++) {
        txtValue = a[i].textContent || a[i].innerText;
        if (txtValue.toUpperCase().indexOf(filter) > -1) {
            a[i].style.display = "";
        } else {
            a[i].style.display = "none";
        }
    }
}

function addMatch(r,sentence){
    let match = r.text()
    console.log(sentence)
    let data = {"action":"addMatch","sentenceID":sentence,"match_desc":match}
    //if(document.getElementById(match) == null){
        r.parent().parent().next('div.technique-scroll').append(`<div class="sentence-tag" id="` + match + 
            `"><button class="close" onclick="deleteMatch('standard')">&#10006;</button><br>` + match + `</div>`)
        
        restRequest('POST',data,refresh)
    /*}else{
        alert("Technique has already been added to the sentence");
    }*/   
}

function deleteMatch(r){
    //if(confirm("Are you sure you want to delete this?")){
        var match = r.parent().text()
        match = match.replace('✖','')
        let sentence = r.parent().next()[0].id
        let data = {"action":"deleteMatch","sentenceID":sentence,"match_desc":match}
        r.parent().remove()
        restRequest('POST',data,refresh)
    //}

}