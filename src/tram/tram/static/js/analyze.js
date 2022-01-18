
var stored_sentences = {}; // stores `GET /api/sentences/` as a dict where {"sentence_id": {sentence}}
var first_sentence_id = -1;
var last_sentence_id = -1;
var active_sentence_id_glob = -1;

$( document ).ready(function() {
    active_sentence_id_glob = 0;

    //$('.select2-use').select2({
    //    placeholder: "Search...",
    //    width: "100%",
    //    tabindex: 1
    //});
    /* var myModal = document.getElementById('addMappingModal')
    var myInput = document.getElementById('myInput')

    myModal.addEventListener('shown.bs.modal', function () {
        myInput.focus()
    }); */

    loadSentences();
});

var lastClick = null;
$(document).keydown(function(e){

    var now = Date.now();
    // Only trigger event if a sentence has been selected, add .4 sentence cooldown
    if ((!lastClick || now - lastClick > 400) && active_sentence_id_glob != -1) {
        lastClick = now;

        // On up arrow, go to prev sentence
        if (e.which == 38 && !e.repeat) { 
            if (active_sentence_id_glob != first_sentence_id){
                loadSentences(String(parseInt(active_sentence_id_glob) - 1));
            }
            return false;
        }
        // On down arrow, go to next sentence
        else if (e.which == 40 && !e.repeat) { 
            if (active_sentence_id_glob != last_sentence_id) {
                loadSentences(String(parseInt(active_sentence_id_glob) + 1));
            }
            return false;
        }
    }
    return false
});

function loadSentences(active_sentence_id) {
    $.ajax({
        type: "GET",
        url: `/api/sentences/?report-id=${REPORT_ID}`,
        dataType: "json",
        success: function (sentences) {
            active_sentence_id_glob = active_sentence_id;
            storeSentences(sentences);
            renderSentences(active_sentence_id);
            renderMappings(active_sentence_id);
        },
        failure: function (data) {
            console.log(`Failure: ${data}`);
        }
    });
}

function storeSentences(sentences) {
    stored_sentences = {};
    var i;
    for (i = 0; i < sentences.length; i++) {
        sentence = sentences[i];
        stored_sentences[sentence.id] = sentence;
        if (i == 0) {
            first_sentence_id = sentence.id
        }
        else if (i == sentences.length - 1) {
            last_sentence_id = sentence.id
        }
    }
}

function renderSentences(active_sentence_id) {
    var $sentenceTable = $(`<table id="sentence-table" class="table table-striped table-hover text-start"><tbody></tbody></table>`)
    for (sentence_id in stored_sentences) {
        sentence = stored_sentences[sentence_id];
        var flag_class = "";
        if (sentence.disposition == null) {
            flag_class = "bg-warning"; // Indicates the sentence is in review
        } else if (sentence.disposition == 'accept') {
            flag_class = "bg-success"; // Indicates the sentence is confirmed
        }
        var numMappings = "";
        if (sentence.mappings.length != 0) {
            numMappings = sentence.mappings.length;
        }

        $row = $(`<tr id="sentence-row-${sentence.id}" style="cursor: pointer;" onclick="renderMappings(${sentence.id})"></tr>`)
        if (sentence.id == active_sentence_id) {
            $row.addClass("bg-info");
        }
        $flag = $("<td style='width: 1%''>"+numMappings+"</td>").addClass(flag_class);
        $text = $(`<td style="word-wrap: break-word;min-width: 160px;max-width: 160px;"></td>`).text(sentence.text);
        $row.append($flag).append($text);
        $sentenceTable.append($row);
    }

    $("#sentence-table").replaceWith($sentenceTable);
}

function renderMappings(sentence_id) {
    active_sentence_id_glob = sentence_id;
    $mappingContainer = $(`<div class="col" id="mapping-container"><h4>Mappings</h4></div>`);

    $("#sentence-table > tbody > tr").removeClass('bg-info')
    if (sentence_id == null) {
        $mappingContainer.append('Select a sentence to see mappings');
        $("#mapping-container").replaceWith($mappingContainer);
        return;
    }

    sentence = stored_sentences[sentence_id];
    mappings = sentence.mappings;

    $mappingTable = $(`<table id="mapping-table" class="table table-striped table-hover"><tbody></tbody></table>`);
    var addButton = `<button class="btn btn-sm btn-success" data-bs-toggle="modal" data-bs-target="#addMappingModal">Add...</button>`;

    $mappingTable.append(`<tr><th>Technique ${addButton}</th><th>Confidence</th><th></th></tr>`);
    for (let i = 0; i < sentence.mappings.length; i++) {
        var mapping = sentence.mappings[i];
        var $row = $(`<tr></tr>`);
        $row.append(`<td>${mapping.attack_id} - ${mapping.name}</td>`);
        $row.append(`<td>${mapping.confidence}%</td>`);
        $removeButton = $(`<button type="button" class="btn btn-sm btn-danger"><i class="fas fa-minus-circle"></i><`);
        $removeButton.click(
            // Need this callback to avoid closure issues when assigning onClick event
            createCallback(sentence.id, mapping.id)
        );
        $row.append($(`<td></td>`).append($removeButton));
        $mappingTable.append($row);
    }
    $mappingContainer.append($mappingTable);
    $dispositionGroup = $(`<div class="btn-group"></div>`);

    var accept_class = accept_text = pending_class = pending_text = "";

    if (sentence.disposition == "accept") {
        accept_class = "btn btn-success";
        review_class = "btn btn-outline-warning";
    } else {
        accept_class = "btn btn-outline-success";
        review_class = "btn btn-warning";
    }

    // Used to determine to go to next sentence on accept click
    // Needed for last sentence edge case
    var next_sentence_id = false;
    if (active_sentence_id_glob == last_sentence_id) {
        next_sentence_id = active_sentence_id_glob;
    }

    var accept_onclick = `updateSentence(${sentence.id}, {disposition: 'accept'}, ${next_sentence_id})`;
    $accept = $(`<button type="button" class="${accept_class}" onclick="${accept_onclick}">Accepted</button>`);

    var review_onclick = `updateSentence(${sentence.id}, {disposition: null}, ${next_sentence_id})`;
    $review = $(`<button type="button" class="${review_class}" onclick="${review_onclick}">Reviewing</button>`);

    $dispositionGroup.append($accept).append($review);
    $mappingContainer.append($dispositionGroup);
    $(`#sentence-row-${sentence.id}`).addClass('bg-info');
    $('#sentence-id').val(sentence.id);

    $("#mapping-container").replaceWith($mappingContainer);
}

function createCallback(sentence_id, mapping_id){
    return function(){
      deleteMapping(sentence_id, mapping_id, false)
    }
  }