var CONFIG;
var news;

function more() {
  loadNews(100);
}

function url_request(api_url, params="") {
  return "http://" + CONFIG.server + ":" + CONFIG.port + api_url + params;
}

function like(index) {
  var li = $(event.srcElement).closest('li');
  n = JSON.parse(JSON.stringify(news[index]));
  n.like = !n.like;
  n.dislike = false;

  if(!news[index].like) { //should be liked
    $.ajax({
      type: "POST",
      url: url_request(CONFIG.API_LIKE_URL),
      data: JSON.stringify(n),
      contentType:"application/json; charset=utf-8"
    })
    .done(function(data) {
      li.removeClass('disliked');
      li.addClass('liked');
      news[index] = n;
    })
    .fail(function(data) {
      alert("Error");
    });
  }
  else {
    li.removeClass('liked');
    news[index] = n;
    alert("notworkingyet");
  }
}

function dislike(index) {
  var li = $(event.srcElement).closest('li');

  if(news[index].dislike) {
    li.removeClass('disliked');
  }
  else {
    li.removeClass('liked');
    li.addClass('disliked');
  }
}

function readArticle(index) {
  var win = window.open(news[index].link, '_blank');
  win.blur();
  window.focus();
  return false;
}

function loadArticlesFromUrl(url, pageSize = 30) {
  $.get(url, function(data) {
    var list = $("#feed");
    list.empty();
    news = data;

    for(var i=0; i<news.length;i++)
    {
      var imgUrl;      
      imgUrl = (news[i].img == "")?"/img/unipi.gif":news[i].img;
      list.append("<li data-index='" + i + "' > <div class='card card-news'><div class='list-header'> <img class='list-img' src='"+ imgUrl + "'></img> <div class='list-category'>"+news[i].source +"</div> <a class='list-title' href='javascript:void(0)' onclick='readArticle("+ i + ")' >" + news[i].title + "</a><div class='list-author'>"+ news[i].author +"</div> <div class='list-datetime'> <i class='fas fa-clock'></i> " + news[i].datetime + "</div><div class='list-content'>"+ news[i].description + "</div> <div class='list-footer'> <i  onclick='like("+ i + ")'  class='far fa-thumbs-up likebtn'></i> | <i  onclick='dislike("+ i + ")'  class='far fa-thumbs-down dislikebtn'></i></div></div></li>");
    }
  });
}

function loadNews(pageSize = 30) {
  var url = url_request(CONFIG.API_FEED_URL, pageSize);
  loadArticlesFromUrl(url, pageSize);
}

function learning(pageSize = 50) {
  var url = url_request(CONFIG.API_LEARN_URL);
  loadArticlesFromUrl(url, pageSize);
}

function loadConfig() {
  $.getJSON('/config/config.json', function(data) {
    CONFIG = data;
    loadNews(30)
  });
}

function man()
{
  $("#searchbar").val('');
  var text = "";// = "<ul>";
  for(var i =0; i < CONFIG.cmds.length; i++)
  {
    text += " - " + CONFIG.cmds[i].name + ": " + CONFIG.cmds[i].description+"\n";
  }
  swal({
    title: "Lista comandi",
    text: text,
    cancel: "Got it"
  });
}

$(document).ready(function() {

  loadConfig();
  $("#searchbar").focus();
  
  $("#search").on('submit', function (e) {
    e.preventDefault();
    var query = $("#searchbar").val();
    var cmd = query.split(" ")[0];
    
    switch(cmd)
    {
      case 'learn':
        $("#searchbar").val('');
        learning();
        break;

      case 'man':
        $("#searchbar").val('');
        man();
        break;
      
      case 'more':
        $("#searchbar").val('');
        more();
        break;
    }
  });

});

$(document).on('keypress', function (e) {
  if(!$("#searchbar").is(":focus") &&
     ![13,38,40,37,39].includes(e.keyCode)) {
    $("#searchbar").val('');
    $("#searchbar").focus();
  }
});